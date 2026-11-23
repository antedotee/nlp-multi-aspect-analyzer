"""Local inference for the ModernBERT-ABSA aspect highlighter."""

from __future__ import annotations

import argparse
import functools
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModelForTokenClassification, AutoTokenizer

DEFAULT_MODEL_PATH = "artifacts/modernbert-absa"
LABELS = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"]

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


def cluster_chips(
    mentions: list[tuple[int, AspectSpan, np.ndarray]],
    threshold: float,
) -> list[AspectChip]:
    """Group (review_id, span, embedding) mentions into aspect chips sorted by frequency."""
    if not mentions:
        return []
    keys = sorted({span.text.strip().lower() for _, span, _ in mentions})
    if len(keys) >= 2:
        centroids = np.stack([
            np.mean([vec for _, span, vec in mentions if span.text.strip().lower() == key], axis=0)
            for key in keys
        ])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
        ).fit_predict(centroids)
    else:
        labels = [0]
    cluster_of = dict(zip(keys, labels))

    groups: dict[int, list[tuple[int, AspectSpan]]] = {}
    for review_id, span, _ in mentions:
        groups.setdefault(cluster_of[span.text.strip().lower()], []).append((review_id, span))

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
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str | None = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device).eval()
        id2label = getattr(self.model.config, "id2label", None) or {}
        if len(id2label) != len(LABELS) or any(k not in id2label for k in range(len(LABELS))):
            id2label = dict(enumerate(LABELS))
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

    @torch.inference_mode()
    def highlight_reviews(
        self,
        reviews: list[str],
        batch_size: int = 16,
        max_length: int = 1024,
        cluster_threshold: float = 0.28,
    ) -> HighlightResult:
        highlights = [ReviewHighlight(id=i, text=r, spans=[]) for i, r in enumerate(reviews)]
        todo = [h for h in highlights if h.text.strip()]
        mentions: list[tuple[int, AspectSpan, np.ndarray]] = []

        for b in range(0, len(todo), batch_size):
            batch = todo[b : b + batch_size]
            enc = self.tokenizer(
                [h.text for h in batch],
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping").tolist()
            mask = enc["attention_mask"].tolist()
            out = self.model(**enc.to(self.device), output_hidden_states=True)
            probs, preds = out.logits.float().softmax(-1).max(-1)
            hidden = out.hidden_states[-1].float().cpu().numpy()
            probs, preds = probs.cpu().tolist(), preds.cpu().tolist()

            for j, h in enumerate(batch):
                n = sum(mask[j])
                if self.tokenizer.padding_side == "left":
                    rows = range(len(mask[j]) - n, len(mask[j]))
                else:
                    rows = range(n)
                tags = [self.id2label[preds[j][t]] for t in rows]
                scores = [probs[j][t] for t in rows]
                offs = [tuple(offsets[j][t]) for t in rows]
                vecs = hidden[j][list(rows)]
                for span, idx in decode_spans(h.text, tags, scores, offs):
                    h.spans.append(span)
                    mentions.append((h.id, span, vecs[idx].mean(axis=0)))

        return HighlightResult(chips=cluster_chips(mentions, cluster_threshold), reviews=highlights)


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
