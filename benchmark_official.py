"""Official-split, multi-seed E2E-ABSA benchmark.

Re-runs the train_and_benchmark.ipynb comparison with the evaluation problems fixed:
  * Data: the original SemEval-2014 Task 4 XML (laptops + restaurants), including the sentences
    with no aspects that the per-aspect Hugging Face copy leaves out.
  * Test: the official gold test sets (800 sentences per domain). Conflict aspects are removed
    from the gold labels; every sentence is kept, as in Li et al. (2019).
  * Train/dev: the official train sets split 90/10 per domain with one fixed split for every run.
    Sentences containing a conflict aspect are dropped from train/dev.
  * Every model trains for the same number of epochs, best epoch picked on dev, over several seeds.
  * Metrics on character spans (tokenizer-independent): strict aspect+polarity P/R/F1 per domain
    and pooled, per-polarity F1, macro-F1, aspect-only F1, and the false-aspect rate on test
    sentences that have no aspects.

Kaggle (T4, internet on):  python benchmark_official.py
Local smoke test:          SMOKE=1 python benchmark_official.py
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4, like the original Colab run (Kaggle has two)

if importlib.util.find_spec("seqeval") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "seqeval"])

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from seqeval.metrics import f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

SMOKE = os.environ.get("SMOKE") == "1"
ON_KAGGLE = Path("/kaggle/working").exists()
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working" if ON_KAGGLE else "benchmark_out"))
RUNS = Path("/tmp/absa_runs")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH_SIZE, WEIGHT_DECAY, WARMUP_RATIO, SPLIT_SEED = 16, 0.01, 0.1, 42

DATA_REV = "3fd43769112ef6517c263e303fbb7dc3002c35df"  # alexcadillon/SemEval2014Task4, pinned
DATA_REPO = "alexcadillon/SemEval2014Task4"
DATA_FILES = {
    ("laptops", "train"): "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml",
    ("restaurants", "train"): "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml",
    ("laptops", "test"): "ABSA_Gold_TestData/Laptops_Test_Gold.xml",
    ("restaurants", "test"): "ABSA_Gold_TestData/Restaurants_Test_Gold.xml",
}
EXPECTED_SENTENCES = {("laptops", "train"): 3045, ("restaurants", "train"): 3041,
                      ("laptops", "test"): 800, ("restaurants", "test"): 800}
DOMAINS = ["laptops", "restaurants"]

LABELS = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
POLARITY = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
CLASSES = ["POS", "NEG", "NEU"]

OURS = "RoBERTa-base + R-Drop"
MODELS = [
    dict(name="BERT-base", ckpt="bert-base-uncased", rdrop=0.0, lr=3e-5, fp16=True),
    dict(name="RoBERTa-base", ckpt="roberta-base", rdrop=0.0, lr=3e-5, fp16=True),
    dict(name="DeBERTa-v3-base", ckpt="microsoft/deberta-v3-base", rdrop=0.0, lr=2e-5, fp16=False),
    dict(name="ModernBERT-base", ckpt="answerdotai/ModernBERT-base", rdrop=0.0, lr=3e-5, fp16=True),
    dict(name=OURS, ckpt="roberta-base", rdrop=1.5, lr=3e-5, fp16=True),
]
if os.environ.get("MODELS"):
    keep = set(os.environ["MODELS"].split(","))
    MODELS = [m for m in MODELS if m["name"] in keep]
if SMOKE:
    MODELS = [m for m in MODELS if m["ckpt"] == "roberta-base"]
    SEEDS, EPOCHS = SEEDS[:1], 1


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ---------------------------------------------------------------- data
def trim(text, s, e):
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def load_xml(domain: str, split: str) -> list[dict]:
    path = Path(hf_hub_download(DATA_REPO, DATA_FILES[(domain, split)], repo_type="dataset", revision=DATA_REV))
    raw = path.read_text(encoding="utf-8")
    if re.search(r"<!(DOCTYPE|ENTITY)", raw):
        raise ValueError(f"{path}: unexpected DTD/entity declarations")
    examples = []
    for sent in ET.fromstring(raw).iter("sentence"):
        text = sent.findtext("text")
        terms = sent.findall("aspectTerms/aspectTerm")
        spans, last_end = [], -1
        for t in sorted(terms, key=lambda t: (int(t.get("from")), int(t.get("to")))):
            if t.get("polarity") not in POLARITY:
                continue
            s, e = trim(text, int(t.get("from")), int(t.get("to")))
            assert text[int(t.get("from")):int(t.get("to"))] == t.get("term"), (path, t.get("term"))
            if s < last_end or s >= e:
                continue
            spans.append((s, e, POLARITY[t.get("polarity")]))
            last_end = e
        examples.append({"text": text, "domain": domain, "spans": spans,
                         "conflict": any(t.get("polarity") == "conflict" for t in terms)})
    assert len(examples) == EXPECTED_SENTENCES[(domain, split)], (domain, split, len(examples))
    return examples


def build_splits():
    rng = random.Random(SPLIT_SEED)
    train, dev, test = [], [], {}
    for d in DOMAINS:
        seen, xs = set(), []
        for x in load_xml(d, "train"):
            if not x["conflict"] and x["text"] not in seen:
                seen.add(x["text"])
                xs.append(x)
        rng.shuffle(xs)
        k = int(0.9 * len(xs))
        train += xs[:k]
        dev += xs[k:]
        test[d] = load_xml(d, "test")
    assert not {x["text"] for x in train} & {x["text"] for x in dev}, "sentence leaked between train and dev"
    if SMOKE:
        train, dev = train[:96], dev[:48]
        test = {d: xs[:60] for d, xs in test.items()}
    return train, dev, test


def describe(train, dev, test) -> dict:
    def row(xs):
        c = Counter(p for x in xs for *_, p in x["spans"])
        return {"sentences": len(xs), "no_aspect": sum(not x["spans"] for x in xs),
                "aspects": sum(c.values()), **{k: c[k] for k in CLASSES}}
    stats = {"train": row(train), "dev": row(dev), **{f"test_{d}": row(xs) for d, xs in test.items()}}
    overlap = {d: len({x["text"] for x in train + dev} & {x["text"] for x in xs}) for d, xs in test.items()}
    stats["train_test_text_overlap"] = overlap
    return stats


# ---------------------------------------------------------------- encoding & metrics
def tag_tokens(offsets, spans):
    tags, prev = [], None
    for s, e in offsets:
        if s == e:
            tags.append(-100)
            continue
        hit = next((sp for sp in spans if s < sp[1] and e > sp[0]), None)
        tags.append(LABEL2ID["O"] if hit is None else LABEL2ID[("I-" if hit == prev else "B-") + hit[2]])
        prev = hit
    return tags


def encode(examples, tok, max_length=256):
    enc = tok([x["text"] for x in examples], truncation=True, max_length=max_length, return_offsets_mapping=True)
    enc["labels"] = [tag_tokens(o, x["spans"]) for o, x in zip(enc.pop("offset_mapping"), examples)]
    return Dataset.from_dict(dict(enc))


def dev_metrics(eval_pred):
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    preds, labels = logits.argmax(-1), eval_pred.label_ids
    y_true = [[ID2LABEL[int(l)] for l in ls if l != -100] for ls in labels]
    y_pred = [[ID2LABEL[int(p)] for p, l in zip(ps, ls) if l != -100] for ps, ls in zip(preds, labels)]
    return {"e2e_f1": f1_score(y_true, y_pred)}


def decode(text, offsets, tag_ids):
    spans, cur = [], None
    for (s, e), t in zip(offsets, tag_ids):
        if s == e:
            continue
        tag = ID2LABEL[t]
        if tag == "O":
            cur = None
            continue
        bio, pol = tag.split("-")
        if bio == "B" or cur is None or cur[2] != pol:
            cur = [s, e, pol]
            spans.append(cur)
        cur[1] = e
    return {(*trim(text, s, e), p) for s, e, p in spans}


@torch.inference_mode()
def predict(model, tok, texts, fp16, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, truncation=True, max_length=512, padding=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        amp = torch.autocast("cuda", dtype=torch.float16) if fp16 and torch.cuda.is_available() else contextlib.nullcontext()
        with amp:
            ids = model(**enc.to(model.device)).logits.argmax(-1).tolist()
        out += [decode(t, o, p) for t, o, p in zip(chunk, offsets, ids)]
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def span_scores(gold: list[set], pred: list[set]) -> dict:
    def count(g_sets, p_sets):
        tp = sum(len(g & p) for g, p in zip(g_sets, p_sets))
        return tp, sum(len(p) for p in p_sets) - tp, sum(len(g) for g in g_sets) - tp
    p, r, f = prf(*count(gold, pred))
    out = {"precision": p, "recall": r, "e2e_f1": f}
    out["ate_f1"] = prf(*count([{x[:2] for x in g} for g in gold], [{x[:2] for x in p} for p in pred]))[2]
    for c in CLASSES:
        out[f"f1_{c}"] = prf(*count([{x for x in g if x[2] == c} for g in gold], [{x for x in p if x[2] == c} for p in pred]))[2]
    out["macro_f1"] = float(np.mean([out[f"f1_{c}"] for c in CLASSES]))
    empty = [bool(p) for g, p in zip(gold, pred) if not g]
    out["false_aspect_rate"] = float(np.mean(empty)) if empty else float("nan")
    return out


# ---------------------------------------------------------------- training
class RDropTokenTrainer(Trainer):
    def __init__(self, *args, rdrop_alpha=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        feats = {k: v for k, v in inputs.items() if k != "labels"}

        def ce(logits):
            return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        out = model(**feats)
        loss = ce(out.logits)
        if model.training and self.rdrop_alpha > 0:
            out2 = model(**feats)
            mask = labels != -100
            p = F.log_softmax(out.logits[mask].float(), -1)
            q = F.log_softmax(out2.logits[mask].float(), -1)
            kl = 0.5 * (F.kl_div(p, q, log_target=True, reduction="batchmean")
                        + F.kl_div(q, p, log_target=True, reduction="batchmean"))
            loss = 0.5 * (loss + ce(out2.logits)) + self.rdrop_alpha * kl
        return (loss, out) if return_outputs else loss


def train_one(cfg, seed, tok, train_ds, dev_ds):
    use_fp16 = cfg["fp16"] and torch.cuda.is_available()
    steps = math.ceil(len(train_ds) / BATCH_SIZE) * EPOCHS
    for fp16 in dict.fromkeys([use_fp16, False]):
        set_seed(seed)
        model = AutoModelForTokenClassification.from_pretrained(
            cfg["ckpt"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, dtype=torch.float32)
        args = TrainingArguments(
            output_dir=str(RUNS / f"{re.sub(r'[^a-z0-9]+', '_', cfg['name'].lower())}_{seed}"),
            learning_rate=cfg["lr"], num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=64, weight_decay=WEIGHT_DECAY, warmup_steps=int(WARMUP_RATIO * steps),
            max_grad_norm=1.0, fp16=fp16, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="e2e_f1", greater_is_better=True,
            logging_steps=100, logging_nan_inf_filter=False, report_to="none", seed=seed,
        )
        trainer = RDropTokenTrainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds, processing_class=tok,
            data_collator=DataCollatorForTokenClassification(tok, pad_to_multiple_of=8),
            compute_metrics=dev_metrics, rdrop_alpha=cfg["rdrop"],
        )
        trainer.train()
        if all(math.isfinite(h["loss"]) for h in trainer.state.log_history if "loss" in h):
            return trainer, fp16, args.output_dir
        log(f"non-finite loss with fp16={fp16}; retrying {cfg['name']} in fp32")
    return trainer, False, args.output_dir


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SMOKE and not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required (set SMOKE=1 for a CPU smoke test).")
    if torch.cuda.is_available():
        log(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    import transformers
    log(f"transformers {transformers.__version__} | seeds {SEEDS} | epochs {EPOCHS} | smoke {SMOKE}")

    train, dev, test = build_splits()
    stats = describe(train, dev, test)
    (OUT / "data_stats.json").write_text(json.dumps(stats, indent=2))
    log("data:", json.dumps(stats))
    gold = {d: [set(x["spans"]) for x in xs] for d, xs in test.items()}

    rows, best_ours = [], -1.0
    for cfg in MODELS:
        tok = AutoTokenizer.from_pretrained(cfg["ckpt"])
        train_ds, dev_ds = encode(train, tok), encode(dev, tok)
        for seed in SEEDS:
            log(f"=== {cfg['name']} | seed {seed}")
            t0 = time.perf_counter()
            trainer, fp16, run_dir = train_one(cfg, seed, tok, train_ds, dev_ds)
            train_min = (time.perf_counter() - t0) / 60
            model = trainer.model.eval()
            preds = {d: predict(model, tok, [x["text"] for x in xs], fp16) for d, xs in test.items()}
            for d in DOMAINS + ["pooled"]:
                g = gold[d] if d != "pooled" else gold["laptops"] + gold["restaurants"]
                p = preds[d] if d != "pooled" else preds["laptops"] + preds["restaurants"]
                rows.append({"model": cfg["name"], "seed": seed, "domain": d, "fp16_train": fp16,
                             "epochs": EPOCHS, "train_min": train_min, "dev_e2e_f1": trainer.state.best_metric,
                             **span_scores(g, p)})
            log({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
            pd.DataFrame(rows).to_csv(OUT / "results_runs.csv", index=False)

            if cfg["name"] == OURS and trainer.state.best_metric > best_ours:
                best_ours = trainer.state.best_metric
                trainer.save_model(OUT / "absa-model")
                (OUT / "absa-model" / "selected_run.json").write_text(json.dumps({"seed": seed, "dev_e2e_f1": best_ours}))
                log(f"saved {OUT / 'absa-model'} (seed {seed}, dev F1 {best_ours:.4f})")

            shutil.rmtree(run_dir, ignore_errors=True)
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    runs = pd.DataFrame(rows)
    metrics = ["e2e_f1", "macro_f1", "ate_f1", "f1_POS", "f1_NEG", "f1_NEU", "false_aspect_rate"]
    summary = runs.groupby(["domain", "model"])[metrics].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(OUT / "results_summary.csv", index=False)

    print("\n=== Strict E2E-F1 on the official test sets: mean ± std over seeds", SEEDS)
    table = runs.groupby(["model", "domain"])["e2e_f1"].agg(["mean", "std"]).unstack("domain")
    for model in [m["name"] for m in MODELS]:
        cells = [f"{100 * table.loc[model, ('mean', d)]:.2f} ± {100 * (table.loc[model, ('std', d)] if len(SEEDS) > 1 else 0):.2f}"
                 for d in DOMAINS + ["pooled"]]
        print(f"  {model:24} laptops {cells[0]:>14} | restaurants {cells[1]:>14} | pooled {cells[2]:>14}")
    print("\n=== Full per-domain summary\n" + summary.to_string())

    if {OURS, "RoBERTa-base"} <= set(runs.model):
        print("\n=== R-Drop minus plain RoBERTa, same seed (strict E2E-F1 points)")
        piv = runs.pivot_table(index=["domain", "seed"], columns="model", values="e2e_f1")
        diff = 100 * (piv[OURS] - piv["RoBERTa-base"])
        for d in DOMAINS + ["pooled"]:
            v = diff.loc[d]
            print(f"  {d:12} per seed {[round(x, 2) for x in v]} | mean {v.mean():+.2f} | wins {(v > 0).sum()}/{len(v)}")
    log("done")


if __name__ == "__main__":
    main()
