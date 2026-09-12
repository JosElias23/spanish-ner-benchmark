# Spanish Named Entity Recognition: from lookup tables to transformers

[![CI](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-73%20passing-brightgreen)](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**English** · [Español](README.es.md)

End-to-end NER for Spanish on the CoNLL-2002 benchmark. Five models, one
evaluation protocol, nothing selected on the held-out test set, and confidence
intervals on every comparison — with a Holm correction over the ten of them.

Try it: `python app/app.py` for an interactive Gradio demo, or `docker compose
-f serve/docker-compose.yml up --build` for the production service with its
latency and cost numbers. Both are described below.

---

## Results

Entity-level F1 on the official held-out test split (`esp.testb`), read exactly
once at the end of the project.

| Model | Params | Test F1 | Test F1 (dedup.) | Precision | Recall | Train time |
|---|---:|---:|---:|---:|---:|---:|
| Gazetteer (longest match) | — | 0.3595 | 0.3364 | 0.2634 | 0.5662 | 0.05 s |
| CRF (hand-engineered features) | — | 0.7924 | 0.7769 | 0.7966 | 0.7881 | 51 s |
| **BETO** (`bert-base-spanish-wwm-cased`) | 109 M | **0.8705** | 0.8642 | 0.8626 | 0.8786 | 105 s |
| **mBERT** (`bert-base-multilingual-cased`) | 177 M | **0.8720** | 0.8659 | 0.8634 | 0.8809 | 110 s |
| XLM-R (`xlm-roberta-base`) | 277 M | 0.8622 | 0.8530 | 0.8568 | 0.8677 | 452 s |

Timings are on a single RTX 5060 Ti (8 GB), four epochs each.

![Model comparison](reports/figures/model_comparison.png)

The *deduplicated* column is explained in [A problem with the
benchmark](#a-problem-with-the-benchmark).

All numbers are produced by `scripts/evaluate_test.py` and stored verbatim in
[`reports/metrics_test.json`](reports/metrics_test.json). Nothing in this README
is typed by hand.

---

## Three findings

### 1. The best model on the development set was the worst on the test set

| Model | Dev F1 | Test F1 | Rank change |
|---|---:|---:|:--|
| XLM-R | **0.8765** (1st) | 0.8622 (3rd) | ▼ 2 |
| mBERT | 0.8665 (2nd) | **0.8720** (1st) | ▲ 1 |
| BETO | 0.8640 (3rd) | 0.8705 (2nd) | ▲ 1 |

Selecting a model on development-set performance alone, the default in most
tutorials, would have shipped the model that finished last on test. This is
model-selection overfitting on a 1,915-sentence dev set, observed directly
rather than described in the abstract.

**What the ranking does not mean.** An earlier version of this section called
XLM-R "the worst of the three transformers" and said bigger and more
multilingual was "strictly worse on every axis that matters". The ranking is
real; the gaps behind it are not distinguishable from noise. Once all ten
pairwise comparisons are corrected for multiplicity (finding 2), no pair of
transformers separates on the test set — XLM-R against BETO is p = 0.30, XLM-R
against mBERT p = 0.13. The honest statement is that XLM-R led dev and finished
last on test by margins this test set cannot resolve.

What survives without qualification is the cost. XLM-R is 277 M parameters and
452 s of training against BETO's 109 M and 105 s — a 4.3× cost for a score that
is, at best, the same.

### 2. A Spanish-specific encoder gives no measurable advantage here

This is the hypothesis the project was built to test, so it is named in the code
as the confirmatory comparison and reported on its own, unadjusted:

| Comparison | ΔF1 | 95% CI | p | Significant |
|---|---:|:--|---:|:--:|
| BETO vs mBERT | −0.0015 | [−0.0113, +0.0086] | 0.75 | **no** |

**BETO and mBERT are statistically indistinguishable, and on this benchmark the
Spanish-specific hypothesis is not supported.** Reporting `mBERT 0.8720 > BETO
0.8705` as a result would have been a claim about sampling noise.

Everything else is exploratory: all ten pairs, in alphabetical order, with a
Holm correction over the family.

| Comparison | ΔF1 | 95% CI | p | p (Holm) | Significant |
|---|---:|:--|---:|---:|:--:|
| BETO vs gazetteer | +0.5110 | [+0.4934, +0.5290] | <0.001 | <0.001 | yes |
| gazetteer vs mBERT | −0.5125 | [−0.5316, −0.4931] | <0.001 | <0.001 | yes |
| gazetteer vs XLM-R | −0.5027 | [−0.5231, −0.4817] | <0.001 | <0.001 | yes |
| CRF vs gazetteer | +0.4329 | [+0.4150, +0.4508] | <0.001 | <0.001 | yes |
| BETO vs CRF | +0.0782 | [+0.0614, +0.0953] | <0.001 | <0.001 | yes |
| CRF vs mBERT | −0.0797 | [−0.0978, −0.0624] | <0.001 | <0.001 | yes |
| CRF vs XLM-R | −0.0698 | [−0.0887, −0.0520] | <0.001 | <0.001 | yes |
| mBERT vs XLM-R | +0.0098 | [+0.0003, +0.0195] | 0.042 | **0.126** | **no** |
| BETO vs XLM-R | +0.0083 | [−0.0028, +0.0207] | 0.151 | 0.303 | no |
| BETO vs mBERT | −0.0015 | [−0.0113, +0.0086] | 0.752 | 0.752 | no |

> **Correction.** This table used to have four rows, all against mBERT, and
> mBERT was chosen because it had the highest test F1:
>
> ```python
> ranked = sorted(results, key=lambda n: results[n]["full"]["overall"]["f1"],
>                 reverse=True)
> for challenger in ranked[1:]:
>     comparisons[f"{ranked[0]}_vs_{challenger}"] = paired_bootstrap(...)
> ```
>
> The reference arm was picked by its score on the same data the p-values come
> from, which biases every comparison toward it, and four tests were printed as
> if each were a single pre-registered one. The row that mattered was **mBERT
> vs XLM-R at p = 0.042, marked "Significant: yes"** — it clears 0.05 by 0.008
> and its interval clears zero by 0.0003 F1. Under Holm it is **0.126**, and it
> is not a finding.
>
> The comparison the ranking-first protocol could never produce is BETO vs
> XLM-R, because neither was the winner. It is p = 0.151, which is why finding 1
> above no longer calls XLM-R the worst of anything.
>
> One thing worth noting in the other direction: the bias ran *against* the
> project's own conclusion. mBERT was the arm selection favoured, and BETO still
> could not be separated from it. Finding 2 was conservative, not flattered.

### 3. Detection is solved; classification is not

![Error breakdown](reports/figures/error_breakdown.png)

Of BETO's 479 test-set errors:

| Error type | Count | Share | What it means |
|---|---:|---:|---|
| Type | 230 | 48.0% | Correct span, wrong class |
| Boundary | 186 | 38.8% | Correct class, wrong span |
| Spurious | 47 | 9.8% | Entity predicted where there is none |
| **Missed** | **16** | **3.3%** | Gold entity not found at all |

The model almost always finds the entity, only 3.3% of errors are outright
misses. Nearly nine out of ten errors are about *labelling* something it already
located. Effort spent on recall would be largely wasted; the payoff is in
disambiguating types, especially ORG vs LOC, and in span boundaries.

![F1 by entity type](reports/figures/f1_by_entity_type.png)

`MISC` is the weakest class for every model (BETO F1 = 0.671 vs 0.958 for
`PER`). That is expected: `MISC` is defined negatively in the annotation
guidelines. It is everything that is an entity but not a person, organisation or
location. So it has no consistent surface form to learn.

The full decision log, including the choices that changed these numbers and
the ones that turned out not to, is in [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## The task

Given Spanish text, tag every mention of a **person**, **organisation**,
**location** or **miscellaneous** entity (nationalities, events, works of art).

```
El   Banco Central de Chile   anuncio hoy en   Santiago
     └────── ORG ─────────┘                    └─ LOC ─┘
```

NER is the extraction layer under document search, compliance screening, KYC and
entity-based analytics. It is also the standard proving ground for sequence
labelling, which makes it a good benchmark for comparing model families under a
fixed protocol.

## The data

[CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) Spanish, from EFE
newswire text (May 2000), annotated by the University of Antwerp.

| Split | File | Sentences | Tokens | Entities |
|---|---|---:|---:|---:|
| Train | `esp.train` | 8,323 | 264,715 | 18,798 |
| Development | `esp.testa` | 1,915 | 52,923 | 4,352 |
| Test | `esp.testb` | 1,517 | 51,533 | 3,559 |

Entities are IOB2-tagged across four types: LOC (4,914), ORG (7,390), PER
(4,321), MISC (2,173) in the training split.

The corpus is parsed from the raw column files by `src/spanish_ner/data.py`
rather than through a hosted loading script, and every file is SHA-256 verified
on load. This matters in practice: `datasets>=3` dropped support for
script-based datasets, which broke the conventional `load_dataset("conll2002",
"es")` path entirely.

> **A parsing detail that cost real F1.** These files are UTF-8, despite this
> corpus almost always being read as latin-1. Decoding them as latin-1 silently
> corrupts about 27,000 accented tokens, `información` becomes `informaciÃ³n`,
> and nothing raises. Training runs, loss decreases, and the score is quietly
> worse. `tests/test_data.py::test_encoding_is_utf8_not_mojibake` is the
> tripwire.

### A problem with the benchmark

The official splits are **not disjoint**. 309 of the 1,517 test sentences
(20.4%) appear verbatim in the training data:

| | Count |
|---|---:|
| Test sentences duplicated in train | 309 (20.4%) |
| ...that are the single token `-` | 162 |
| ...that are agency datelines (`Madrid , 23 may ( EFE ) .`) | ~52 |
| ...with 10 or more tokens (real content) | 86 |
| **Test entities inside duplicated sentences** | **336 / 3,559 (9.4%)** |

Most of the overlap is newswire boilerplate, but 9.4% of test entities sit in
sentences the model has already seen. Removing them from the official benchmark
would break comparability with published results, so every table reports both:

- **Full**: the official 1,517 sentences, comparable with the literature.
- **Deduplicated**: the 1,166 *distinct* sentences absent from training, a
  stricter estimate of generalisation to genuinely unseen text.

  That number used to read 1,208 here, which is the count after removing test
  sentences that occur in training but before removing the 42 that repeat
  within the test set itself. `scripts/evaluate_test.py` has always dropped
  both — a sentence of boilerplate appearing four times should not be weighted
  four times in a memorisation-free measurement — but only the first step was
  described, so the prose named a set 42 sentences larger than the one that was
  scored. `reports/metrics_test.json` now records all three counts.

The ranking is unchanged, but the *cost* of deduplication is not uniform, and
the pattern is informative:

| Model | Full | Deduplicated | Drop |
|---|---:|---:|---:|
| Gazetteer | 0.3595 | 0.3364 | **−2.31 pts** |
| CRF | 0.7924 | 0.7769 | −1.55 pts |
| XLM-R | 0.8622 | 0.8530 | −0.92 pts |
| BETO | 0.8705 | 0.8642 | −0.63 pts |
| mBERT | 0.8720 | 0.8659 | −0.61 pts |

The drop is ordered exactly by overall F1, and that is the problem with the
reading an earlier version of this section gave it.

> **Correction.** This used to say: "the more a model relies on memorisation,
> the more it loses when memorised sentences are removed — the gazetteer, which
> is nothing but memorisation, gives up almost four times as much as the
> transformers", and called it a direct measurement of generalisation.
>
> Two checks break it, and both use only the numbers already on this page.
>
> **The drop ordering is the F1 ordering, reversed, exactly.** Spearman
> correlation between overall test F1 and drop across the five models is
> **−1.00**. A perfect rank correlation across five models is the signature of
> an arithmetic constraint, not of a behavioural difference.
>
> **The constraint is headroom.** Removing the duplicated sentences removes
> 9.4% of test entities. Treating micro-F1 as approximately a weighted average
> over the two portions, a model that scored *perfectly* on the removed 9.4%
> could drop at most `0.094 × (1 − F1) / 0.906` when they go:
>
> | Model | Test F1 | Largest drop arithmetic allows | Observed |
> |---|---:|---:|---:|
> | Gazetteer | 0.3595 | 6.64 pts | 2.31 |
> | CRF | 0.7924 | 2.15 pts | 1.55 |
> | XLM-R | 0.8622 | 1.43 pts | 0.92 |
> | BETO | 0.8705 | 1.34 pts | 0.63 |
> | mBERT | 0.8720 | 1.33 pts | 0.61 |
>
> The ceiling ratio between gazetteer and mBERT is **5.0×** before any
> behaviour enters. The observed ratio is **3.8×** — *below* the mechanical
> bound. A low-scoring model has more room to fall, and these drops are
> entirely consistent with that and with nothing else.
>
> What the table does show is smaller and still worth keeping: every model loses
> something, so every model was getting some credit from sentences it had seen
> in training, and the deduplicated column is the more honest estimate. What it
> does not show is which model relied on memorisation more. Separating that
> would need each model's score *on the duplicated sentences* compared against
> like-for-like sentences absent from training, and that experiment is not run
> here.

These counts are pinned in `tests/test_data.py::TestSplitOverlap`, so if the raw
data ever changes the test suite fails rather than the metrics drifting.

---

## Approach

Four modelling families, in increasing order of capacity. Each exists to make
the next one's score interpretable.

**1. Gazetteer.** Memorise every entity surface form in the training set;
longest-match at inference; resolve type ambiguity by majority vote. This is the
floor, and it quantifies how much of the task is pure memorisation (F1 = 0.36.
About 41% of BETO's score with no learning at all).

**2. CRF.** A linear-chain conditional random field over hand-designed features:
casing, prefixes and suffixes up to three characters, digit and hyphen patterns,
the PAROLE POS tag both full and truncated to two characters, and all of the
above for a ±2 token window. Suffix features carry more signal in Spanish than
in English because of its richer inflectional morphology. A CRF scores the whole
tag sequence jointly, so it learns structural constraints such as *I-PER never
follows B-ORG* directly from the transition weights. This was state of the art
before contextual embeddings, and it is the real bar a transformer has to clear.

**3–5. Transformers.** BETO, mBERT and XLM-R fine-tuned for token
classification, all through the same script so no result can differ because of
an accidental change in procedure.

### Sub-word label alignment

The central technical problem. The corpus is annotated per word; BERT tokenises
into sub-words:

```
words       ["Telefónica",  "invirtió"          ]
labels      ["B-ORG",       "O"                 ]
sub-tokens  ["Telef", "##ónica", "invir", "##tió"]
```

Labels must be re-projected onto sub-tokens. We label the **first** sub-token of
each word and assign `-100` to the rest; PyTorch's cross-entropy ignores `-100`,
so continuation sub-tokens contribute no loss and no gradient.

The alternative, repeating the label on every sub-token, over-weights long words
in the loss and makes decoding ambiguous when sub-tokens of the same word
disagree. It also requires converting `B-` to `I-` on continuations, or the
decoded sequence contains two adjacent `B-ORG` tags and splits one entity into
two. Both strategies are implemented; `--label-all-subtokens` runs the ablation.

Misaligned labels are the classic silent failure in NER fine-tuning: the model
trains, the loss falls, and the target was wrong the whole time. Ten tests in
`tests/test_modeling.py` pin the alignment contract, including one that asserts
the test word actually splits into sub-tokens. Otherwise the other tests would
pass while proving nothing.

### Evaluation protocol

- **Entity-level scoring** via `seqeval`, matching the official CoNLL script. A
  prediction counts only if type *and* both boundaries are exact. Token-level
  accuracy is meaningless here: ~88% of tokens are outside any entity, so a
  model predicting `O` everywhere scores 0.88 accuracy while finding nothing.
- **Nothing is selected on the test split.** Baseline selection,
  hyperparameters and early stopping all run on dev, and reading `esp.testb`
  requires passing `include_test=True` so it cannot happen by accident. Two
  scripts pass it: `scripts/evaluate_test.py`, which produces the reported
  scores, and `scripts/error_analysis.py`, which describes the errors of the
  already-chosen model and changed none of them. An earlier version of this
  bullet said `evaluate_test.py` was the only file that read the test split,
  which was not true.
- **Best epoch by dev F1 is kept**, not the last, so a model that peaks at epoch
  3 and overfits at epoch 4 is not reported at its worst.
- **Reported scores are recomputed through `predict_sentences`**, the same
  inference path the demo uses, rather than trusting the `Trainer`'s internal
  loop. If the two ever disagree, the deployed model is not the one measured.
- **Paired bootstrap** on 10,000 resamples for every comparison. Both models are
  scored on the same resampled sentences, controlling for the fact that some
  sentences are simply harder. The implementation resamples precomputed
  per-sentence TP/predicted/gold counts instead of re-running `seqeval`, which
  turns hours into milliseconds.
- **Fixed seed (42)** across `random`, `numpy` and `torch`.

---

## Reproducing these numbers

Every command below was run to produce the tables above. Total wall-clock time
on an RTX 5060 Ti (8 GB): **about 15 minutes** for all five models plus
evaluation.

```bash
git clone https://github.com/JosElias23/spanish-ner-benchmark.git
cd spanish-ner-benchmark
```

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
```

Or skip straight to `make all`, which runs the whole pipeline end to end in the
correct order.

```bash
python scripts/download_data.py
```

Downloads the three raw corpus files and verifies their SHA-256 checksums.

```bash
python -m pytest
```

73 tests: corpus integrity, encoding, split overlap, entity decoding, sub-word
label alignment, the significance protocol and the service contract. Run this before trusting any number
below. The API tests skip themselves when no checkpoint has been trained yet, so
a fresh clone runs green.

```bash
python scripts/train_baselines.py
```

Gazetteer and CRF, evaluated on dev. About one minute.

```bash
python scripts/train_transformer.py
python scripts/train_transformer.py --model bert-base-multilingual-cased --tag mbert
python scripts/train_transformer.py --model xlm-roberta-base --tag xlmr
```

BETO and mBERT take under two minutes each on a modern consumer GPU; XLM-R takes
about 7.5 minutes. CPU works but is considerably slower.

```bash
python scripts/evaluate_test.py \
    --models models/bert-base-spanish-wwm-cased/best models/mbert/best models/xlmr/best
```

The single read of the held-out test set. Writes `reports/metrics_test.json`.

```bash
python scripts/error_analysis.py
```

Regenerates every figure in this README plus `reports/error_analysis.json`.

```bash
python app/app.py
```

Serves the Gradio demo at `http://127.0.0.1:7860`.

---

## Serving: latency, throughput and cost

A demo proves the model runs. These are the numbers an operator needs before
putting it behind anything.

Measured on a single RTX 5060 Ti. **Every row processes the same 512
development-set documents** (mean 24.2 tokens), in chunks of the stated batch
size, three passes each after five discarded warm-ups. p50/p95/p99 are per-batch
latencies across every chunk of every pass; throughput is the whole 512-document
workload divided by the median pass time. Produced by `serve/benchmark.py` and
stored in [`reports/metrics_serving.json`](reports/metrics_serving.json).

| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Docs/s | USD per 1M docs |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.4 | 7.1 | 7.6 | 160 | 0.92 |
| 4 | 8.9 | 12.1 | 12.5 | 431 | 0.34 |
| 8 | 13.7 | 20.5 | 22.2 | 551 | 0.27 |
| 16 | 24.5 | 36.7 | 37.2 | 638 | 0.23 |
| **32** | 52.8 | 64.3 | 64.7 | **638** | **0.23** |
| 64 | 112.7 | 127.7 | 128.4 | 605 | 0.24 |

**Batching 32 documents delivers 4.0× the throughput of one-at-a-time and cuts
cost per million documents by 75%.**

> **Correction.** This table used to read 6.7× and put the optimum at batch 64,
> because each row was timed on `pool[:batch_size]` — batch 1 on one particular
> sentence, batch 64 on a different set of 64. The cross-row comparison was
> therefore a mixture of batch size and which documents happened to be
> measured, and the batch-1 row was the latency of a single unlucky sentence
> rather than a median. With the workload held fixed the speed-up is 4.0×, and
> **throughput peaks at 32 and falls at 64** — so the old advice to put a bulk
> pipeline at the right-hand end of the table pointed past the optimum.

The shape is the point. Going from batch 1 to batch 4 costs 2.6 ms of latency
and nearly triples throughput, because a batch of four barely fills the GPU.
From 16 to 32 throughput does not move at all while latency doubles, and at 64
it goes backwards. The device saturates around 16, and past that batching buys
queueing delay. An interactive endpoint belongs at the left of this table, a
bulk pipeline in the middle, and neither number alone describes the system.

**About the cost column.** The rate is USD 0.53/hour for an NVIDIA T4 and the
throughput is this machine's RTX 5060 Ti, which is considerably faster. So the
column is a rate times a measured throughput, not a quote: a real T4 would bill
more per million documents. The mismatch is recorded in
`reports/metrics_serving.json` as `priced_hardware` and `measured_hardware`
rather than left for a reader to infer. An earlier version of this section said
only "an entry-level inference GPU", which gave no way to notice.

### Running it

```bash
pip install -e ".[serve]"
python serve/benchmark.py                    # reproduce the GPU table above
python serve/benchmark.py --device cpu       # the CPU table, for the container
uvicorn serve.api:app --port 8000            # serve locally
docker compose -f serve/docker-compose.yml up --build
```

The service exposes `POST /extract`, separate `/health` and `/ready` probes, and
`/metrics` in Prometheus text format with rolling p50/p95/p99 over the last
10,000 requests. Liveness and readiness are deliberately distinct: weights take
seconds to load, and an orchestrator routing on liveness alone would send
traffic to a pod that cannot answer.

```bash
curl -s localhost:8000/extract -H 'content-type: application/json' \
  -d '{"texts":["El Banco Central de Chile anuncio hoy en Santiago."]}'
```

The container is multi-stage and runs as a non-root user on CPU-only torch,
which keeps the image small enough for a free tier.

> **Correction.** This sentence used to end "while still clearing 100 documents
> per second at batch 64", and **no CPU benchmark existed in this repository**:
> `reports/metrics_serving.json` was a single CUDA run. `serve/benchmark.py`
> always had a `--device cpu` flag that had never been used to produce a
> committed report. It has now, into
> [`reports/metrics_serving_cpu.json`](reports/metrics_serving_cpu.json):
>
> | Batch | p50 (ms) | Docs/s | USD per 1M docs |
> |---:|---:|---:|---:|
> | 1 | 28.7 | 35.0 | 0.79 |
> | **4** | 82.0 | **48.6** | **0.57** |
> | 8 | 162.9 | 47.8 | 0.58 |
> | 16 | 368.7 | 43.2 | 0.64 |
> | 32 | 888.0 | 38.9 | 0.71 |
> | 64 | 2005.8 | 34.5 | 0.80 |
>
> **48.6 documents per second at batch 4, not 100 at batch 64** — and batch 64
> is the worst setting on the table, no better than one-at-a-time. Batching is
> worth 1.4× on this CPU against 4.0× on the GPU, which is the more useful
> finding: the CPU is compute-bound at batch 1 already, so grouping work buys
> latency and almost no throughput. The free-tier deployment is viable at a few
> dozen documents per second, and the earlier number was an unmeasured guess
> that happened to be threefold optimistic.

Inference in the service runs through `predict_sentences`, the same function
that produced every metric in this README, and a test asserts that batched and
unbatched extraction return identical entities. Padding a short sequence beside
a long one is exactly where an attention-mask bug hides, and it degrades output
rather than raising.

---

## Deploying the demo

The trained checkpoint is not committed. It is 440 MB and regenerable in under
two minutes by `scripts/train_transformer.py`. Publishing the demo therefore
means publishing the weights first.

```bash
huggingface-cli login
huggingface-cli upload JosElias23/spanish-ner-beto \
    models/bert-base-spanish-wwm-cased/best
```

Then create a Space at
[huggingface.co/new-space](https://huggingface.co/new-space) with SDK **Gradio**
and hardware **CPU basic (free)**, and give it this layout:

```
app.py              <- app/app.py, unchanged
requirements.txt    <- app/requirements.txt, unchanged
spanish_ner/        <- the whole src/spanish_ner package
```

Copying only the two files is not enough. The app imports `predict_sentences`
and `iter_entities` from the package on purpose, so that the demo runs the same
inference code that produced the reported metrics, and the package has to travel
with it. At the Space root, `spanish_ner/` is importable directly; the
`sys.path` line in `app.py` resolves to nothing there and is simply inert.

Finally set the Space variable `NER_MODEL_PATH` to `JosElias23/spanish-ner-beto`.

That variable is the only thing that differs between environments: the app reads
it and falls back to the local checkpoint path when it is unset, so exactly the
same file runs on a laptop and in the Space.

BETO is the deployed model rather than mBERT. The two are statistically
indistinguishable (ΔF1 = 0.0015, p = 0.75), and given a tie the Spanish-specific
model is the better default for Spanish input: a smaller, better-fitted
vocabulary means fewer sub-tokens per word and faster CPU inference.

---

## Repository layout

```
spanish-ner-benchmark/
├── configs/default.yaml        every hyperparameter that affects a metric
├── data/raw/                   corpus files (gitignored, checksum-verified)
├── src/spanish_ner/
│   ├── data.py                 CoNLL parser, label schema, overlap analysis
│   ├── baselines.py            gazetteer and CRF
│   ├── modeling.py             sub-word alignment, transformer inference
│   ├── evaluate.py             seqeval scoring, paired bootstrap
│   └── utils.py                seeding, config, JSON reporting
├── scripts/
│   ├── download_data.py        fetch and verify the corpus
│   ├── train_baselines.py      gazetteer + CRF, dev evaluation
│   ├── train_transformer.py    fine-tune any HF checkpoint
│   ├── evaluate_test.py        the reported scores; reads the test split
│   └── error_analysis.py       figures and error categorisation
├── serve/
│   ├── api.py                  FastAPI service with probes and metrics
│   ├── benchmark.py            latency, throughput and cost measurement
│   └── Dockerfile              multi-stage, non-root, CPU-only
├── tests/                      73 tests
├── reports/                    metrics as JSON, figures as PNG
└── app/                        Gradio demo for Hugging Face Spaces
```

---

## Limitations and next steps

Stated plainly, because every one of these is a question worth being asked.

**Domain.** Anecdotally the model transfers better than the training data
suggests: it correctly tags `Gabriel Boric` and `Rosanna Costa` as PER and
`Banco Central de Chile` as ORG, none of which can appear in a Spanish corpus
from the year 2000. That is encouraging, and it is **not evidence**. It is four
hand-picked sentences with no gold annotations behind them.

The corpus is Spanish newswire from May 2000, from a single agency (EFE), and
heavily European in vocabulary and place names. Performance on Chilean text,
social media, clinical notes or legal documents will be substantially lower and
is **not measured here**. Any number in this README is a claim about EFE
newswire and nothing else.

**MISC is weak.** F1 = 0.671, against 0.958 for PER. The class is defined
negatively in the annotation guidelines, so it has no consistent surface form.
Improving it likely needs a different formulation, not more training data.

**Single seed.** Each configuration was trained once with seed 42. The bootstrap
quantifies uncertainty from the *test sample*, not from training-run variance.
Seed variance for BERT fine-tuning on a corpus this size is typically a few
tenths of an F1 point. Comparable to the BETO/mBERT gap, which reinforces
finding 2 rather than undermining it. Training each model across five seeds and
reporting mean ± standard deviation is the correct next step and is not done
here.

**No hyperparameter search.** Learning rate, batch size and epoch count are
standard values from the BERT fine-tuning literature, applied identically to all
three transformers. Fair for comparison, almost certainly not optimal for any of
them.

**Truncation.** `max_length=256` sub-tokens, and it does bite. An earlier
version of this bullet said "no training sentence is affected, so this costs
nothing on this corpus" — contradicted by this repository's own committed
reports, which record **6 truncated training sentences and 1,531 lost words**
for BETO (6 / 1,537 for mBERT, 5 / 1,449 for XLM-R). Six sentences out of 8,323
sounds negligible until you notice they are the long ones: 1,531 of 264,715
training words, **0.58%**. Small, but not the nothing that was claimed. The report is written by
`scripts/train_transformer.py` and covers the training split only — the dev and
test splits are not checked, so nothing is known about truncation there. The
demo also silently tags words beyond the limit as `O`; long documents should be
chunked before being passed in.

**The dev/test rank inversion deserves more work.** Finding 1 is a single
observation. Whether XLM-R is genuinely worse here or was unlucky needs repeated
runs to settle.

### Planned

- Multi-seed training with variance reporting
- Evaluation on Chilean Spanish to measure the domain gap directly
- A CRF decoding layer on top of the transformer, which should reduce the 38.8%
  boundary-error share
- Confidence calibration, so the demo can flag uncertain predictions

---

## Citation

```bibtex
@inproceedings{tjongkimsang2002conll,
  title     = {Introduction to the CoNLL-2002 Shared Task: Language-Independent
               Named Entity Recognition},
  author    = {Tjong Kim Sang, Erik F.},
  booktitle = {Proceedings of CoNLL-2002},
  year      = {2002}
}

@inproceedings{canete2020spanish,
  title     = {Spanish Pre-Trained BERT Model and Evaluation Data},
  author    = {Ca{\~n}ete, Jos{\'e} and Chaperon, Gabriel and Fuentes, Rodrigo and
               Ho, Jou-Hui and Kang, Hojin and P{\'e}rez, Jorge},
  booktitle = {PML4DC at ICLR 2020},
  year      = {2020}
}
```

## License

**MIT, see [LICENSE](LICENSE).** The CoNLL-2002 corpus is distributed by the
University of Antwerp under its own terms.