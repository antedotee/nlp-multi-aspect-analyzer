"""Score aspect-chip grouping against hand-labelled gold groups (tests/grouping_gold.json).

Compares the original grouping (the tagger's embedding layer on raw phrases, threshold 0.22)
with the current one (normalize_aspect + sentence embeddings) across thresholds.

    python eval_grouping.py [--gold tests/grouping_gold.json] [--model-path artifacts/absa-model]
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

from infer import DEFAULT_CLUSTER_THRESHOLD, DEFAULT_MODEL_PATH, get_grouper, normalize_aspect


def load_gold(path: str) -> tuple[list[str], list[int]]:
    with open(path, encoding="utf-8") as f:
        gold = json.load(f)
    phrases, labels = [], []
    for g, group in enumerate(gold["groups"]):
        phrases += group
        labels += [g] * len(group)
    for k, phrase in enumerate(gold["singletons"]):
        phrases.append(phrase)
        labels.append(len(gold["groups"]) + k)
    return phrases, labels


def cluster(keys: list[str], vectors: np.ndarray, threshold: float) -> dict[str, int]:
    if len(keys) < 2:
        return {k: 0 for k in keys}
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
    ).fit_predict(vectors)
    return dict(zip(keys, labels))


def score(phrases: list[str], gold: list[int], pred: list[int]) -> dict:
    tp = fp = fn = 0
    wrong, missed = [], []
    for i, j in itertools.combinations(range(len(phrases)), 2):
        same_gold, same_pred = gold[i] == gold[j], pred[i] == pred[j]
        tp += same_gold and same_pred
        if same_pred and not same_gold:
            fp += 1
            wrong.append(f"{phrases[i]} + {phrases[j]}")
        if same_gold and not same_pred:
            fn += 1
            missed.append(f"{phrases[i]} / {phrases[j]}")
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "ari": adjusted_rand_score(gold, pred), "wrong": wrong, "missed": missed}


@torch.inference_mode()
def old_vectors(phrases: list[str], model_path: str) -> np.ndarray:
    """The original method: mean embedding-layer vector of the phrase's tokens inside a sentence."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).eval()
    vecs = []
    for phrase in phrases:
        text = f"The {phrase} is great."
        s, e = 4, 4 + len(phrase)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        hidden = model(**enc, output_hidden_states=True).hidden_states[0][0]
        idx = [i for i, (a, b) in enumerate(offsets) if a < e and b > s and a != b]
        v = hidden[idx].mean(0).numpy()
        vecs.append(v / np.linalg.norm(v))
    return np.stack(vecs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", default="tests/grouping_gold.json")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = ap.parse_args()
    phrases, gold = load_gold(args.gold)
    n_groups = len(set(gold))
    print(f"{len(phrases)} phrases in {n_groups} gold chips\n")

    rows = []
    old_keys = sorted({p.strip().lower() for p in phrases})
    old_vec = dict(zip(old_keys, old_vectors(old_keys, args.model_path)))
    assign = cluster(old_keys, np.stack([old_vec[k] for k in old_keys]), 0.22)
    rows.append(("old: embedding layer @0.22", score(phrases, gold, [assign[p.strip().lower()] for p in phrases])))

    grouper = get_grouper()
    if grouper is None:
        raise SystemExit("grouping model unavailable (offline?)")
    new_keys = sorted({normalize_aspect(p) for p in phrases})
    new_vec = grouper.embed(new_keys)
    for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        assign = cluster(new_keys, new_vec, t)
        mark = "  <- default" if abs(t - DEFAULT_CLUSTER_THRESHOLD) < 1e-9 else ""
        rows.append((f"new: normalize + MiniLM @{t:.2f}{mark}", score(phrases, gold, [assign[normalize_aspect(p)] for p in phrases])))

    print(f"{'method':42} {'precision':>9} {'recall':>7} {'F1':>6} {'ARI':>6}  wrong merges")
    for name, s in rows:
        print(f"{name:42} {s['precision']:9.2f} {s['recall']:7.2f} {s['f1']:6.2f} {s['ari']:6.2f}  {len(s['wrong'])}")
    for name, s in (rows[0], next(r for r in rows if "default" in r[0])):
        print(f"\n{name}\n  wrong merges: {s['wrong'] or 'none'}\n  missed merges ({len(s['missed'])}): {s['missed'][:12]}{' ...' if len(s['missed']) > 12 else ''}")


if __name__ == "__main__":
    main()
