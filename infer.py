"""Local inference for the RoBERTa-ABSA (R-Drop) aspect highlighter."""

from __future__ import annotations

import argparse
import functools
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

# The tagger: a local copy if one has been trained or downloaded, otherwise the published weights.
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "absa-model"
HF_MODEL_ID = "Deepanshu027/roberta-base-rdrop-absa-semeval14"
DEFAULT_MODEL_PATH = str(LOCAL_MODEL_DIR) if (LOCAL_MODEL_DIR / "config.json").is_file() else HF_MODEL_ID
LABELS = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"]

# Chip grouping: a small sentence-embedding model groups aspect phrases by meaning.
# The tagger's own embedding layer only captures spelling, so it cannot merge synonyms.
GROUPER_MODEL = os.environ.get("ABSA_GROUPER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROUPER_TEMPLATE = "The product's {}."
DEFAULT_CLUSTER_THRESHOLD = 0.25
ATTRIBUTE_WORDS = {"life", "quality", "level"}
DETERMINERS = {"the", "a", "an", "my", "this", "that", "these", "those", "its", "their", "our", "your"}
PLURAL_EXCEPTIONS = {"series", "species", "lens", "news", "windows", "macos", "ios", "chromeos", "plus", "bus"}

log = logging.getLogger(__name__)

Sentiment = Literal["POS", "NEG", "NEU"]


@dataclass
class AspectSpan:
    text: str
    start: int
    end: int
    sentiment: Sentiment
    score: float


@dataclass
class ReviewHighlight:
    id: int
    text: str
    spans: list[AspectSpan]


@dataclass
class AspectChip:
    aspect: str
    pos: int
    neg: int
    neu: int
    total: int
    review_ids: list[int]


@dataclass
class HighlightResult:
    chips: list[AspectChip]
    reviews: list[ReviewHighlight]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_spans(
    review: str,
    tags: list[str],
    scores: list[float],
    offsets: list[tuple[int, int]],
) -> list[tuple[AspectSpan, list[int]]]:
    """Turn per-token BIO tags into character spans on `review`, plus the token indices of each span."""
    spans: list[tuple[AspectSpan, list[int]]] = []
    current: tuple[str, list[int]] | None = None

    def close() -> None:
        if current is None:
            return
        sentiment, idx = current
        start, end = offsets[idx[0]][0], offsets[idx[-1]][1]
        while start < end and review[start].isspace():
            start += 1
        while end > start and review[end - 1].isspace():
            end -= 1
        if start < end:
            score = float(np.mean([scores[i] for i in idx]))
            spans.append((AspectSpan(review[start:end], start, end, sentiment, score), idx))

    for i, tag in enumerate(tags):
        if offsets[i][0] == offsets[i][1]:
            continue
        prefix, _, sentiment = tag.partition("-")
        if prefix == "I" and (current is None or current[0] != sentiment):
            prefix = "B"
        if prefix == "B":
            close()
            current = (sentiment, [i])
        elif prefix == "I":
            current[1].append(i)
        else:
            close()
            current = None
    close()
    return spans


_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)|\n")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character (start, end) of each sentence or line in `text`, whitespace trimmed, covering all words."""
    cuts, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        cuts.append((start, m.end()))
        start = m.end()
    cuts.append((start, len(text)))
    spans = []
    for s, e in cuts:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            spans.append((s, e))
    return spans


def _singular(word: str) -> str:
    if len(word) <= 3 or not word.endswith("s") or word in PLURAL_EXCEPTIONS:
        return word
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    return word[:-1]


def normalize_aspect(text: str) -> str:
    """Grouping key for an aspect phrase: 'The Battery Life' -> 'battery', 'speakers' -> 'speaker'.

    Only used to decide which chip a mention belongs to; chips still display the original wording.
    """
    key = re.sub(r"\s+", " ", text.replace("’", "'").strip().lower()).strip(" .,:;!?\"'()[]“”")
    words = key.split()
    if not words:
        return key
    while len(words) > 1 and words[0] in DETERMINERS:
        words.pop(0)
    if len(words) > 1 and words[-1] in ATTRIBUTE_WORDS:
        words.pop()
    words[-1] = _singular(words[-1])
    return " ".join(words)


class AspectGrouper:
    """Embeds aspect phrases with a small sentence-embedding model (mean pooling, unit length)."""

    def __init__(self, model_name: str = GROUPER_MODEL, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def embed(self, phrases: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(phrases), batch_size):
            enc = self.tokenizer(
                [GROUPER_TEMPLATE.format(p) for p in phrases[i : i + batch_size]],
                padding=True, truncation=True, max_length=64, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


_GROUPERS: dict[str, AspectGrouper] = {}


def get_grouper(model_name: str = GROUPER_MODEL) -> AspectGrouper | None:
    """Load the grouping model once; return None (exact-key grouping) if it cannot be loaded, e.g. offline.

    Failures are not cached, so a transient network error does not disable grouping for the whole session.
    """
    if model_name not in _GROUPERS:
        try:
            _GROUPERS[model_name] = AspectGrouper(model_name)
        except Exception as exc:  # network or cache failure must not break highlighting
            log.warning("Aspect grouper %r unavailable (%s); chips will group by exact normalized phrase.", model_name, exc)
            return None
    return _GROUPERS[model_name]


def cluster_chips(
    mentions: list[tuple[int, AspectSpan]],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    grouper: AspectGrouper | None = None,
) -> list[AspectChip]:
    """Group (review_id, span) mentions into aspect chips sorted by frequency.

    Mentions are keyed by normalize_aspect(); keys are clustered by cosine distance of their
    grouper embeddings (average linkage). Without a grouper, each normalized key is its own chip.
    """
    if not mentions:
        return []
    keys = sorted({normalize_aspect(span.text) for _, span in mentions})
    if len(keys) >= 2 and grouper is not None:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
        ).fit_predict(grouper.embed(keys))
    else:
        labels = range(len(keys))
    cluster_of = dict(zip(keys, labels))

    groups: dict[int, list[tuple[int, AspectSpan]]] = {}
    for review_id, span in mentions:
        groups.setdefault(cluster_of[normalize_aspect(span.text)], []).append((review_id, span))

    chips = []
    for members in groups.values():
        counts = Counter(span.text.strip().lower() for _, span in members)
        name = min(counts, key=lambda s: (-counts[s], len(s), s))
        sentiments = Counter(span.sentiment for _, span in members)
        chips.append(AspectChip(
            aspect=name,
            pos=sentiments["POS"],
            neg=sentiments["NEG"],
            neu=sentiments["NEU"],
            total=len(members),
            review_ids=sorted({review_id for review_id, _ in members}),
        ))
    return sorted(chips, key=lambda c: (-c.total, c.aspect))


class AspectHighlighter:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str | None = None,
                 grouper_model: str | None = GROUPER_MODEL):
        self.device = device or pick_device()
        self.grouper_model = grouper_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device).eval()
        id2label = getattr(self.model.config, "id2label", None) or {}
        if len(id2label) != len(LABELS) or any(k not in id2label for k in range(len(LABELS))):
            id2label = dict(enumerate(LABELS))
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def _segments(self, text: str, max_len: int) -> list[tuple[int, int]]:
        """Split `text` into chunks the model can read whole, cutting only between sentences.

        A review that fits stays one chunk (identical to tagging it directly). Longer reviews are
        packed sentence by sentence up to the limit; a single over-long sentence is cut into token windows.
        """
        limit = max_len - 2 - 8  # special tokens, plus slack for re-tokenisation at chunk edges
        start, end = text.find(text.strip()[:1]), len(text.rstrip())
        # verbose=False: counting tokens of a long text is intended here, not an overflow.
        if len(self.tokenizer(text[start:end], add_special_tokens=False, verbose=False)["input_ids"]) <= limit:
            return [(start, end)]

        pieces = []
        for s, e in sentence_spans(text):
            offs = self.tokenizer(text[s:e], add_special_tokens=False, return_offsets_mapping=True,
                                  verbose=False)["offset_mapping"]
            if len(offs) <= limit:
                pieces.append((s, e, len(offs)))
                continue
            cuts = [0] + [offs[i][0] for i in range(limit, len(offs), limit)] + [e - s]
            pieces += [(s + a, s + b, limit) for a, b in zip(cuts, cuts[1:]) if text[s + a : s + b].strip()]

        chunks, cur = [], None
        for s, e, n in pieces:
            if cur and cur[2] + n + 1 <= limit:
                cur = (cur[0], e, cur[2] + n + 1)
            else:
                if cur:
                    chunks.append(cur[:2])
                cur = (s, e, n)
        chunks.append(cur[:2])
        return chunks

    @torch.inference_mode()
    def highlight_reviews(
        self,
        reviews: list[str],
        batch_size: int = 16,
        max_length: int = 512,
        cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    ) -> HighlightResult:
        highlights = [ReviewHighlight(id=i, text=r, spans=[]) for i, r in enumerate(reviews)]
        mentions: list[tuple[int, AspectSpan]] = []
        model_max_length = getattr(self.tokenizer, "model_max_length", max_length)
        if model_max_length < 100_000:
            effective_max_len = min(max_length, model_max_length)
        else:
            effective_max_len = max_length

        # A review that fits the model is tagged whole; a longer one is split between sentences
        # (see _segments), so no text is ever truncated away.
        segments = [(h, s, e) for h in highlights if h.text.strip()
                    for s, e in self._segments(h.text, effective_max_len)]

        for b in range(0, len(segments), batch_size):
            batch = segments[b : b + batch_size]
            enc = self.tokenizer(
                [h.text[s:e] for h, s, e in batch],
                return_offsets_mapping=True,
                truncation=True,
                max_length=effective_max_len,
                padding=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping").tolist()
            mask = enc["attention_mask"].tolist()
            out = self.model(**enc.to(self.device))
            probs, preds = out.logits.float().softmax(-1).max(-1)
            probs, preds = probs.cpu().tolist(), preds.cpu().tolist()

            for j, (h, s, e) in enumerate(batch):
                n = sum(mask[j])
                if self.tokenizer.padding_side == "left":
                    rows = range(len(mask[j]) - n, len(mask[j]))
                else:
                    rows = range(n)
                tags = [self.id2label[preds[j][t]] for t in rows]
                scores = [probs[j][t] for t in rows]
                offs = [tuple(offsets[j][t]) for t in rows]
                for span, _ in decode_spans(h.text[s:e], tags, scores, offs):
                    span = AspectSpan(span.text, span.start + s, span.end + s, span.sentiment, span.score)
                    h.spans.append(span)
                    mentions.append((h.id, span))

        grouper = get_grouper(self.grouper_model) if self.grouper_model and len(mentions) > 1 else None
        return HighlightResult(chips=cluster_chips(mentions, cluster_threshold, grouper), reviews=highlights)


@functools.lru_cache(maxsize=4)
def get_highlighter(model_path: str = DEFAULT_MODEL_PATH) -> AspectHighlighter:
    return AspectHighlighter(model_path)


def highlight_reviews(reviews: list[str], model_path: str = DEFAULT_MODEL_PATH, **kwargs) -> dict:
    return asdict(get_highlighter(model_path).highlight_reviews(reviews, **kwargs))


SAMPLE_REVIEWS = [
    "Battery life is incredible, easily two days, but the charger feels cheap and flimsy.",
    "The screen is bright and sharp. Shipping took forever though.",
    "",
    "Sound quality is decent for the price, nothing special. The battery drains fast when streaming.",
    "Customer service replaced my unit quickly. Love the build quality and the camera.",
    "Setup was confusing and the app keeps crashing, but the speakers sound amazing.",
]


def check(result: HighlightResult, reviews: list[str]) -> None:
    n_spans = 0
    for h in result.reviews:
        assert h.text == reviews[h.id]
        for s in h.spans:
            assert 0 <= s.start < s.end <= len(h.text), s
            assert s.text == h.text[s.start : s.end], s
            assert s.sentiment in {"POS", "NEG", "NEU"}, s
            assert 0.0 <= s.score <= 1.0, s
            n_spans += 1
    assert sum(c.total for c in result.chips) == n_spans
    assert all(c.total == c.pos + c.neg + c.neu for c in result.chips)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highlight aspects in sample Amazon reviews.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    result = get_highlighter(args.model_path).highlight_reviews(SAMPLE_REVIEWS)
    print("Customers say:")
    for c in result.chips:
        print(f"  {c.aspect:<24} +{c.pos} -{c.neg} ~{c.neu}  reviews={c.review_ids}")
    for h in result.reviews:
        print(f"\n[{h.id}] {h.text!r}")
        for s in h.spans:
            print(f"    {s.sentiment} {s.score:.2f} [{s.start}:{s.end}] {s.text!r}")
    check(result, SAMPLE_REVIEWS)
    print("\nself-check passed")
