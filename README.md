# Multi-Aspect Sentiment Analyzer

An Amazon-style **"Customers say"** highlighter for product reviews. It finds every product aspect a review mentions, labels each one positive, negative or neutral, and groups related aspects across many reviews into clickable summary chips.

> *"The **battery life** is incredible, but the **charging case** feels cheap."*
> → `battery life · positive`, `charging case · negative`

The model is a RoBERTa-base token tagger fine-tuned with **R-Drop** on SemEval-2014. On the official SemEval-2014 test sets it reaches **68.9 F1 on laptops and 78.9 on restaurants** (strict aspect + sentiment match, mean of 3 seeds). It beats plain RoBERTa on every seed.

**Trained weights:** [🤗 Deepanshu027/roberta-base-rdrop-absa-semeval14](https://huggingface.co/Deepanshu027/roberta-base-rdrop-absa-semeval14)

---

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Training and benchmarking](#training-and-benchmarking)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [References](#references)

---

## How it works

```mermaid
flowchart LR
    A[Reviews] --> B[Split long reviews<br/>between sentences]
    B --> C[RoBERTa + R-Drop<br/>token tagger]
    C --> D[Decode BIO tags<br/>to character spans]
    D --> E[Normalise phrases +<br/>MiniLM embeddings]
    E --> F[Agglomerative<br/>clustering]
    F --> G[Aspect chips<br/>+ highlighted reviews]
```

**1. Joint extraction and sentiment.** End-to-end aspect-based sentiment analysis (E2E-ABSA) is framed as sequence labelling. Each token gets one of 7 tags, so a single forward pass finds aspects and their sentiment together:

```
The   battery  life   is  great  but  the  charging  case   feels  cheap
O     B-POS    I-POS  O   O      O    O    B-NEG     I-NEG  O      O
```

Tags: `O`, plus `B-`/`I-` (first token / continuation) for each of `POS`, `NEG` and `NEU`.

**2. R-Drop regularisation.** During training, each batch goes through the model twice with different dropout masks. A symmetric KL term penalises the two predictions for disagreeing:

```
loss = ½(CE₁ + CE₂) + α · ½(KL(p₁‖p₂) + KL(p₂‖p₁)),   α = 1.5
```

This roughly doubles training compute and adds nothing at inference.

**3. Exact highlights.** Predicted tags are mapped back to character offsets in the original text, so the UI highlights exactly the words the model found. Each span's confidence is its mean token probability.

**4. Long reviews.** A review that fits the model's 512-token window is tagged in one pass. Longer reviews are packed into chunks that break only between sentences; a single over-long sentence is cut into token windows. No text is ever dropped.

**5. Aspect chips.** Mentions are grouped by meaning:
- **Clean up each phrase.** `"The Battery life"` → `battery`, `"speakers"` → `speaker`. The original wording is kept for display.
- **Embed it** with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) inside a short template (`"The product's {aspect}."`).
- **Cluster** with average-linkage agglomerative clustering at cosine distance 0.25.

Synonyms such as *price / cost* or *audio / sound* end up in one chip, while *screen* and *screen protector* stay apart. On a 68-phrase held-out set, grouping F1 is **0.87**, against 0.20 for the original embedding-layer approach. If the grouping model can't be downloaded (e.g. offline), chips fall back to exact matches on the cleaned phrase.

---

## Results

### Official SemEval-2014 test sets

Strict E2E-F1: a prediction counts only if the character span **and** the sentiment both match exactly. Values are mean ± std over seeds 42, 43 and 44. Every model trains for 6 epochs, with the best epoch picked on dev. The test sets have 800 sentences per domain, including those with no aspects.

| Model | Laptops | Restaurants | Both |
|---|---|---|---|
| **RoBERTa-base + R-Drop** | **68.89 ± 0.20** | **78.90 ± 0.25** | **75.29 ± 0.21** |
| DeBERTa-v3-base | 68.26 ± 0.82 | 78.61 ± 1.33 | 74.84 ± 1.03 |
| RoBERTa-base | 67.47 ± 0.81 | 77.86 ± 0.56 | 74.11 ± 0.48 |
| ModernBERT-base | 59.45 ± 1.35 | 74.17 ± 0.47 | 68.86 ± 0.76 |
| BERT-base | 59.71 ± 0.31 | 73.24 ± 0.64 | 68.34 ± 0.48 |

- **R-Drop vs RoBERTa:** R-Drop beat plain RoBERTa on all 3 seeds in both domains, by +1.42 points on laptops and +1.04 on restaurants.
- **R-Drop vs DeBERTa-v3:** R-Drop is ahead on average, but by less than DeBERTa's seed-to-seed spread, so treat it as a tie. R-Drop trains in about half the time (3.9 vs 7.3 minutes per seed on a T4).
- **By sentiment** (R-Drop, both test sets): positive F1 0.82, negative 0.71, neutral 0.55.
- **False aspects:** on test sentences with no aspect, the model still predicts one 4.8% of the time on laptops and 9.3% on restaurants.

Per-run results, the summary and the full log are in [`artifacts/official_run/`](artifacts/official_run/).

### Compared with published work

| Method | Laptops | Restaurants |
|---|---|---|
| ChatGPT, zero-shot (Zhang et al., 2023) | 33.1 | 54.5 |
| BERT-base (Li et al., 2019) | 60.4–61.1 | 73.2–74.7* |
| BARTABSA (Yan et al., 2021) | 67.4 | 73.6 |
| GAS, T5-base (Zhang et al., 2021) | 68.1 | 77.1 |
| GRACE (Luo et al., 2020) | 70.7 | 77.3* |
| InstructABSA (Scaria et al., 2023) | 79.3 | 79.5 |
| **This project: BERT-base** | 59.7 | 73.2 |
| **This project: RoBERTa + R-Drop** | 68.9 | 78.9 |

\* SemEval 2014–2016 restaurant data combined, not Rest14 alone.

**Caveat:** published systems average 5–10 seeds and usually train one model per domain. This project trains a single model on both domains.

> **Note:** `train_and_benchmark.ipynb` is the original exploration notebook. It evaluates on a random 10% slice of the *training* data with one seed, so its numbers in `artifacts/benchmark_results.csv` are **not** comparable with the table above. Use `benchmark_official.py` for the official evaluation.

---

## Quick start

Requires **Python 3.10+**. A GPU is optional for inference: the code uses CUDA, Apple Silicon (MPS) or CPU automatically.

```bash
git clone https://github.com/antedotee/nlp-multi-aspect-analyzer.git
cd nlp-multi-aspect-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Run it:**

```bash
python infer.py   # tags sample reviews and runs a self-check
python app.py     # opens the web UI at http://127.0.0.1:7860
```

No training is needed. The trained weights are on the [Hugging Face Hub](https://huggingface.co/Deepanshu027/roberta-base-rdrop-absa-semeval14). If there's no local model in `artifacts/absa-model/`, they download automatically on first run (about 500 MB, then cached). The chip-grouping model (`all-MiniLM-L6-v2`, about 90 MB) downloads the same way.

**Optional: keep a local copy of the weights**, e.g. for offline use:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Deepanshu027/roberta-base-rdrop-absa-semeval14', local_dir='artifacts/absa-model')"
```

**Optional: train your own model.** This trains only the model the app uses and takes about 5 minutes on a T4 GPU (see [Training and benchmarking](#training-and-benchmarking)):

```bash
MODELS="RoBERTa-base + R-Drop" SEEDS=42 python benchmark_official.py
cp -r benchmark_out/absa-model artifacts/absa-model
```

A local model in `artifacts/absa-model/` always takes priority over the Hub copy.

---

## Usage

### Web app

Run `python app.py`, paste reviews one per line, and click **Analyze Reviews**:
- Aspects are highlighted by sentiment; hover over one to see its confidence.
- Chips summarise each aspect across all reviews, for example `battery (2+ / 1-)`.
- Clicking a chip filters the reviews to those that mention it.

### Python API

```python
from infer import highlight_reviews

result = highlight_reviews(["Battery life is great, but the case feels flimsy."])
```

```python
{
  "chips": [
    {"aspect": "battery", "pos": 1, "neg": 0, "neu": 0, "total": 1, "review_ids": [0]},
    {"aspect": "case",    "pos": 0, "neg": 1, "neu": 0, "total": 1, "review_ids": [0]}
  ],
  "reviews": [
    {"id": 0, "text": "Battery life is great, but the case feels flimsy.",
     "spans": [
       {"text": "Battery life", "start": 0,  "end": 12, "sentiment": "POS", "score": 0.998},
       {"text": "case",         "start": 31, "end": 35, "sentiment": "NEG", "score": 0.991}
     ]}
  ]
}
```

For repeated calls, `get_highlighter()` returns a cached `AspectHighlighter`. Its `highlight_reviews(reviews, batch_size=16, max_length=512, cluster_threshold=0.25)` returns typed dataclasses instead of dicts.

---

## Training and benchmarking

`benchmark_official.py` trains and evaluates all five encoders under one protocol:

| | |
|---|---|
| **Data** | Original SemEval-2014 Task 4 XML (laptops + restaurants), pinned to a fixed Hugging Face revision. Includes the sentences with no aspects. |
| **Train / dev** | Official training sets, split 90/10 per domain with one fixed split (5,346 / 595 sentences). Sentences containing a `conflict` aspect are dropped. |
| **Test** | Official gold test sets, 800 sentences per domain. Conflict spans are unlabelled; every sentence is kept. |
| **Models** | BERT-base, RoBERTa-base, DeBERTa-v3-base, ModernBERT-base, RoBERTa-base + R-Drop |
| **Training** | 6 epochs, batch 16, weight decay 0.01, 10% warmup, gradient clip 1.0; best epoch chosen by dev F1. fp16, except DeBERTa-v3, which uses fp32. |
| **Metrics** | Strict span + sentiment P/R/F1 per domain and combined; per-sentiment F1; macro-F1; span-only F1; false-aspect rate |

It needs a CUDA GPU. The full run (5 models × 3 seeds) takes about one hour of training on a single T4.

```bash
python benchmark_official.py                        # full benchmark
SEEDS=42,43,44,45,46 python benchmark_official.py   # more seeds
SMOKE=1 python benchmark_official.py                # tiny CPU run to check the pipeline
```

**On Kaggle:** create a *Script* notebook from `benchmark_official.py`, select a **GPU T4** accelerator, and turn **Internet on**. Results and the selected model are written to `/kaggle/working/`. The script limits itself to one GPU, so results match a single-T4 setup.

**Outputs:**
- `results_runs.csv`: one row per model × seed × domain.
- `results_summary.csv`: mean and std per model and domain.
- `data_stats.json`: split statistics.
- `absa-model/`: the R-Drop run with the best dev F1.

---

## Testing

```bash
pip install pytest
python -m pytest tests -q     # 41 tests
python eval_grouping.py       # chip-grouping precision / recall / F1 by threshold
```

- **Grouping and normalisation:** `tests/test_grouping.py` covers synonym merges, pairs that must stay separate, chip counts and the offline fallback.
- **Long reviews:** `tests/test_long_reviews.py` covers sentence splitting, reviews over 512 tokens, single over-long sentences and offset correctness.
- Tests that need the trained model or the grouping model are skipped automatically when those aren't available.

---

## Project structure

```
├── app.py                      # Gradio web UI
├── infer.py                    # Inference: tagging, long-review chunking, span decoding, chip grouping
├── benchmark_official.py       # Official-split, multi-seed training and evaluation of 5 encoders
├── eval_grouping.py            # Scores chip grouping against a labelled gold set
├── train_and_benchmark.ipynb   # Original exploration notebook (random split, single seed)
├── requirements.txt
├── tests/
│   ├── test_grouping.py
│   ├── test_long_reviews.py
│   └── grouping_gold.json      # Held-out aspect groups for grouping evaluation
└── artifacts/
    ├── absa-model/             # Optional local copy of the weights (git-ignored)
    ├── official_run/           # Official benchmark results and logs
    ├── kaggle_run/             # Reproduction of the original notebook
    ├── benchmark_results.csv   # Original notebook results
    └── pareto_plot.png         # Original notebook accuracy-vs-speed plot
```

---

## Configuration

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `ABSA_MODEL_PATH` | `app.py` | `artifacts/absa-model` if present, else the Hub model | Local folder or Hugging Face repo id of the tagger |
| `ABSA_GROUPER_MODEL` | `infer.py` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for chip grouping |
| `SEEDS` | `benchmark_official.py` | `42,43,44` | Comma-separated training seeds |
| `EPOCHS` | `benchmark_official.py` | `6` | Epochs for every model |
| `MODELS` | `benchmark_official.py` | all five | Comma-separated model names to run |
| `OUT_DIR` | `benchmark_official.py` | `benchmark_out` (`/kaggle/working` on Kaggle) | Output folder |
| `SMOKE` | `benchmark_official.py` | unset | `1` runs a tiny CPU pipeline check |

---

## Limitations

- **Neutral sentiment is the weakest class:** F1 0.55, against 0.82 for positive.
- **Contrast and idioms:** sentences that set up an expectation and then reverse it ("expected it to be terrible, but it is excellent"), and idioms such as "dies fast", can get the wrong sentiment with high confidence.
- **Domain shift:** the model is trained on laptop and restaurant reviews only. It hasn't been evaluated on a labelled set of other product reviews (e.g. headphones), and individual errors do occur.
- **False aspects:** the model invents an aspect in roughly 5–9% of sentences that mention none.
- **Model comparison:** R-Drop vs DeBERTa-v3 is not settled with 3 seeds. ModernBERT used the shared learning rate (3e-5), which is likely too low for it.
- **Chip threshold:** the grouping threshold was chosen on a small, hand-labelled phrase set. Borderline pairs such as *earbuds / ear cushions* can merge.

---

## References

- Pontiki et al. 2014. *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.* SemEval.
- Li, Bing, Zhang & Lam 2019. *Exploiting BERT for End-to-End Aspect-based Sentiment Analysis.* W-NUT @ EMNLP.
- Liang et al. 2021. *R-Drop: Regularized Dropout for Neural Networks.* NeurIPS.
- Liu et al. 2019. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
- He, Gao & Chen 2023. *DeBERTaV3.* ICLR.
- Warner et al. 2024. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT).*
- Reimers & Gurevych 2019. *Sentence-BERT.* EMNLP.

SemEval-2014 Task 4 data is distributed for research use; check the original task terms before redistributing it or any model trained on it.
