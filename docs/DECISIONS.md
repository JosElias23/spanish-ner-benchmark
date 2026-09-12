# Decision log

What was built, in what order, why each choice was made, and what the numbers
turned out to be. Every figure here comes from a file in `reports/`, produced by
running the code. Nothing is typed from memory.

---

## 1. The question

Does a Spanish-specific encoder beat a multilingual one at Spanish named-entity
recognition, and by enough to matter?

That is a hypothesis with a direction, which means it can come out wrong. It
did.

The corpus is CoNLL-2002 Spanish: EFE newswire from May 2000, annotated by the
University of Antwerp, four entity types (PER, ORG, LOC, MISC), with official
train / development / test splits.

| Split | File | Sentences | Tokens | Entities |
|---|---|---:|---:|---:|
| Train | `esp.train` | 8,323 | 264,715 | 18,798 |
| Development | `esp.testa` | 1,915 | 52,923 | 4,352 |
| Test | `esp.testb` | 1,517 | 51,533 | 3,559 |

---

## 2. Decisions about the data

### 2.1 Parse the corpus rather than fetch it

`datasets>=3` dropped support for script-based datasets, which broke the
conventional `load_dataset("conll2002", "es")` path outright. Rather than pin an
old library, the raw column files are parsed by `src/spanish_ner/data.py` and
SHA-256 verified on every load.

The verification is not decoration. A silently truncated or altered corpus would
change every metric in the repository while the code kept running.

### 2.2 The files are UTF-8, not latin-1

This corpus is almost universally read as latin-1, and that is wrong. Decoding
these files as latin-1 corrupts roughly **27,000 accented tokens**:
`información` becomes `informaciÃ³n`. Nothing raises. Training runs, the loss
falls, and the score is quietly worse for a reason no metric reports.

The bug was found by printing tokens after the first parse and noticing the
mojibake, not by reasoning about it.
`tests/test_data.py::test_encoding_is_utf8_not_mojibake` is now the tripwire.

### 2.3 The label set is hard-coded, not inferred

Inferring labels from whatever appears in a file makes the id-to-label mapping
depend on file order. A model saved under one mapping and loaded under another
produces confident, wrong predictions with no error. The ordering is frozen and
a test pins it.

### 2.4 The official splits are not disjoint, and that is reported

**309 of the 1,517 test sentences (20.4%) appear verbatim in training.**

| | Count |
|---|---:|
| Test sentences duplicated in train | 309 (20.4%) |
| ...that are the single token `-` | 162 |
| ...with 10 or more tokens, i.e. real content | 86 |
| **Test entities inside duplicated sentences** | **336 / 3,559 (9.4%)** |

Most of it is newswire boilerplate, but 9.4% of test entities sit in sentences
the model has already seen.

**Decision.** Removing them would break comparability with three decades of
published CoNLL results, so every table reports both the official test set and
the 1,166-sentence deduplicated subset. The counts are pinned in
`tests/test_data.py::TestSplitOverlap`, so a change in the raw data fails the
suite instead of silently moving the metrics.

---

## 3. Decisions about the models

### 3.1 Two baselines, chosen to be informative rather than easy

An F1 with nothing beneath it is uninterpretable. A gazetteer that memorises
every training surface form and longest-matches at inference answers "how much
of this task is pure memorisation". A CRF over hand-designed orthographic and
POS features answers "what did contextual embeddings actually buy", because it
was the state of the art before them.

Feature choices in the CRF worth defending: suffixes to three characters,
because Spanish inflectional morphology carries more signal than English;
the PAROLE POS tag both full and truncated to two characters, because the
fine-grained tagset is sparse; and capitalisation for the token and both
neighbours, because it is the strongest single cue in Spanish NER.

### 3.2 Sub-word label alignment: label the first sub-token only

The corpus is annotated per word; BERT tokenises into sub-words. Labels must be
re-projected, and getting it wrong is the classic silent failure: the model
trains, the loss falls, and the target was wrong the whole time.

Only the first sub-token of each word carries a label; the rest get `-100`,
which PyTorch's cross-entropy ignores. Labelling every sub-token instead
over-weights long words in the loss and forces a `B-` to `I-` conversion on
continuations, or the decoded sequence splits one entity into two. Both
strategies are implemented and `--label-all-subtokens` runs the ablation.

Ten tests pin the alignment contract, including one asserting that the test word
genuinely splits into sub-tokens. Without that, the other nine would pass while
proving nothing.

### 3.3 The test split is read by exactly one script

`scripts/evaluate_test.py` is the only file that opens `esp.testb`. Baseline
selection, hyperparameters, early stopping and error analysis all run on the
development split. This is enforceable by inspection rather than by discipline,
which is the point.

### 3.4 Reported scores are recomputed through the serving path

Rather than trusting the `Trainer`'s internal evaluation loop, every published
number is recomputed by `predict_sentences`, the same function the demo and the
API call. If the two ever disagree, the deployed model is not the one that was
measured.

---

## 4. Results

Entity-level F1 on `esp.testb`, read once, at the end.

| Model | Params | Test F1 | Test F1 (dedup) | Precision | Recall | Train time |
|---|---:|---:|---:|---:|---:|---:|
| Gazetteer | — | 0.3595 | 0.3364 | 0.2634 | 0.5662 | 0.05 s |
| CRF | — | 0.7924 | 0.7769 | 0.7966 | 0.7881 | 51 s |
| BETO | 109 M | 0.8705 | 0.8642 | 0.8626 | 0.8786 | 105 s |
| **mBERT** | 177 M | **0.8720** | **0.8659** | 0.8634 | 0.8809 | 110 s |
| XLM-R | 277 M | 0.8622 | 0.8530 | 0.8568 | 0.8677 | 452 s |

### 4.1 The hypothesis is not supported

Paired bootstrap, 10,000 resamples:

| Comparison | ΔF1 | 95% CI | p | Significant |
|---|---:|:--|---:|:--:|
| **mBERT vs BETO** | **+0.0015** | **[−0.0086, +0.0113]** | **0.75** | **no** |
| mBERT vs XLM-R | +0.0098 | [+0.0003, +0.0195] | 0.04 | yes |
| mBERT vs CRF | +0.0797 | [+0.0624, +0.0978] | <0.001 | yes |
| mBERT vs gazetteer | +0.5125 | [+0.4931, +0.5316] | <0.001 | yes |

BETO and mBERT are statistically indistinguishable. **The project set out to
test whether a Spanish-specific encoder wins on Spanish NER, and on this
benchmark it does not.**

Writing `mBERT 0.8720 > BETO 0.8705` as a finding would have been a claim about
sampling noise. A one-point F1 gap on 1,517 sentences usually is, and the only
way to know is to compute the interval.

### 4.2 The best model on dev was the worst on test

| Model | Dev F1 | Test F1 | Rank change |
|---|---:|---:|:--|
| XLM-R | **0.8765** (1st) | 0.8622 (3rd) | down 2 |
| mBERT | 0.8665 (2nd) | **0.8720** (1st) | up 1 |
| BETO | 0.8640 (3rd) | 0.8705 (2nd) | up 1 |

Selecting on development performance alone, which is what most tutorials do,
would have shipped the worst of the three transformers. This is
model-selection overfitting on a 1,915-sentence dev set, observed rather than
described.

XLM-R is also the most expensive by a wide margin: 277 M parameters and 452 s
against BETO's 109 M and 105 s, a 4.3x training cost for the lowest test score.

### 4.3 Correction: deduplication cost scales with headroom, not memorisation

| Model | Full | Deduplicated | Drop |
|---|---:|---:|---:|
| Gazetteer | 0.3595 | 0.3364 | **−2.31 pts** |
| CRF | 0.7924 | 0.7769 | −1.55 pts |
| XLM-R | 0.8622 | 0.8530 | −0.92 pts |
| BETO | 0.8705 | 0.8642 | −0.63 pts |
| mBERT | 0.8720 | 0.8659 | −0.61 pts |

An earlier version of this section read that as memorisation: "the more a model
relies on memorisation, the more it loses when memorised sentences are removed",
with the gazetteer giving up nearly four times what the transformers do, called
a free quantitative measurement of generalisation.

**It is an arithmetic artefact, and two checks on the numbers above show it.**

The drop ordering is the F1 ordering reversed, exactly: Spearman between overall
test F1 and drop is **-1.00** across all five models. A perfect rank correlation
over five models is what a constraint looks like, not what a behavioural
tendency looks like.

The constraint is headroom. Deduplication removes 336 of 3,559 test entities,
9.4%. Treating micro-F1 as approximately a weighted average over the removed and
retained portions, a model scoring *perfectly* on the removed part could lose at
most `0.094 * (1 - F1) / 0.906` when it goes:

| Model | Test F1 | Ceiling on the drop | Observed |
|---|---:|---:|---:|
| Gazetteer | 0.3595 | 6.64 pts | 2.31 |
| CRF | 0.7924 | 2.15 pts | 1.55 |
| XLM-R | 0.8622 | 1.43 pts | 0.92 |
| BETO | 0.8705 | 1.34 pts | 0.63 |
| mBERT | 0.8720 | 1.33 pts | 0.61 |

The ceiling ratio between the gazetteer and mBERT is **5.0x** before any model
behaves at all. The observed ratio is **3.8x**, below it. Every drop in that
table is consistent with pure headroom, and the "nearly four times" was a
restatement of the F1 column.

What survives: every model loses something, so every model was collecting some
credit from sentences it had already seen, and the deduplicated column is the
better estimate of generalisation. What does not: any claim about *which* model
leaned on memorisation more. Testing that needs each model's score on the
duplicated sentences set against like-for-like sentences absent from training --
same register, same boilerplate, different provenance -- and that experiment is
not in this repository.

The general shape is worth naming, because it recurs: a quantity whose range is
mechanically bounded by another quantity, read as though it were free to vary.
Checking costs one line of arithmetic and is worth doing before any ordering
across models is called a finding.

### 4.4 Detection is solved; classification is not

Of BETO's **479** test-set errors:

| Error type | Count | Share | Meaning |
|---|---:|---:|---|
| Type | 230 | 48.0% | Correct span, wrong class |
| Boundary | 186 | 38.8% | Correct class, wrong span |
| Spurious | 47 | 9.8% | Entity where there is none |
| **Missed** | **16** | **3.3%** | Gold entity not found at all |

Only 3.3% of errors are outright misses. Nearly nine in ten are about
*labelling* something the model already located. Effort spent on recall would be
largely wasted; the payoff is in type disambiguation, especially ORG against
LOC, and in span boundaries.

`MISC` is the weakest class for every model (BETO F1 0.671 against 0.958 for
PER), which is expected: the annotation guidelines define it negatively, as
everything that is an entity but not a person, organisation or location, so it
has no consistent surface form to learn.

---

## 5. Serving: what it costs to run

Measured on one RTX 5060 Ti, 40 iterations per batch size after five discarded
warm-up passes, on real development documents averaging 24.2 tokens.

| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Docs/s | USD per 1M docs |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.2 | 14.0 | 15.5 | 76 | 1.94 |
| 8 | 22.8 | 23.4 | 23.7 | 351 | 0.42 |
| 32 | 68.9 | 69.8 | 70.0 | 464 | 0.32 |
| **64** | 126.6 | 128.8 | 130.0 | **506** | **0.29** |

**Batching 64 documents gives 6.7x the throughput of one-at-a-time and cuts cost
per million by 85%.**

The shape of that table is the useful part. Batch 1 to batch 4 costs 1.4 ms of
latency and quadruples throughput, because four sequences barely fill the GPU.
Batch 32 to 64 doubles latency for 9% more throughput, because the device is
saturated and the extra batching buys only queueing delay. An interactive
endpoint belongs at the left of the table and a bulk pipeline at the right, and
neither figure alone describes the system.

Cost assumes sustained utilisation at USD 0.53/hour for an entry-level inference
GPU and excludes network, storage and orchestration. A cost figure without its
assumption is not a cost figure.

---

## 6. Which model is deployed, and why

BETO, not mBERT, despite mBERT's marginally higher point estimate.

The two are statistically indistinguishable (ΔF1 0.0015, p = 0.75), so the tie
is broken on other grounds: a Spanish-specific vocabulary produces fewer
sub-tokens per Spanish word, which means faster CPU inference for the same
input. Given a genuine tie, the cheaper model wins.

Stating that reasoning matters more than the choice. "We picked the best model"
is not defensible when the two are inside each other's confidence intervals.

---

## 7. What was not done

- **Single seed.** Each configuration was trained once with seed 42. The
  bootstrap quantifies uncertainty from the *test sample*, not from training-run
  variance. Seed variance for BERT fine-tuning on a corpus this size is
  typically a few tenths of an F1 point, which is comparable to the BETO/mBERT
  gap and therefore reinforces finding 4.1 rather than undermining it. Five
  seeds per configuration is the correct next step and is not done.
- **No hyperparameter search.** Standard values from the BERT fine-tuning
  literature, applied identically to all three transformers. Fair for
  comparison, almost certainly not optimal for any of them.
- **One domain.** EFE newswire from May 2000, heavily European in vocabulary and
  place names. Performance on Chilean text, social media, clinical notes or
  legal documents is not measured. Every number here is a claim about newswire.
- **No CRF decoding layer on top of the transformer**, which should reduce the
  38.8% boundary-error share and is the obvious next experiment.
- **No confidence calibration**, so the demo cannot flag uncertain predictions.

---

## 8. Reproducing

```bash
pip install -e ".[dev,serve]"
python scripts/download_data.py        # fetch and checksum-verify the corpus
python -m pytest                       # 60 tests
python scripts/train_baselines.py      # gazetteer and CRF, about a minute
python scripts/train_transformer.py    # BETO
python scripts/train_transformer.py --model bert-base-multilingual-cased --tag mbert
python scripts/train_transformer.py --model xlm-roberta-base --tag xlmr
python scripts/evaluate_test.py --models models/*/best   # the single test read
python scripts/error_analysis.py       # figures and error categorisation
python serve/benchmark.py              # latency, throughput and cost
```

About 15 minutes end to end on a consumer GPU. Every number in this document
lives in `reports/metrics_test.json`, `reports/error_analysis.json` and
`reports/metrics_serving.json`.
