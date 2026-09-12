"""Final evaluation on the held-out test split (esp.testb).

This is the only script in the repository that loads the test set. Everything
else -- baseline selection, hyperparameters, early stopping, error analysis --
happens on the development split. Running this script is the last step, and its
output is what the README reports.

Two numbers are produced for every model:

  full        the official 1,517-sentence test set, comparable with published
              CoNLL-2002 results
  deduplicated the 1,166 *distinct* test sentences that do not appear verbatim
              in training, a stricter estimate of generalisation to unseen text.
              1,208 test sentences are absent from training; 42 of those are
              repeats of each other, and the index below drops those too. An
              earlier version of this docstring and of the README called the
              column 1,208, which is the count before the second
              deduplication and not the count that was scored.

Usage:
    python scripts/evaluate_test.py --models models/bert-base-spanish-wwm-cased/best
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The comparison this project exists to make, named here rather than derived
# from the results. Choosing it after seeing a score is the defect this constant
# prevents.
CONFIRMATORY = ("bert-base-spanish-wwm-cased", "mbert")

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from spanish_ner.baselines import CRFTagger, GazetteerTagger  # noqa: E402
from spanish_ner.data import deduplicate_against, load_all  # noqa: E402
from spanish_ner.evaluate import (  # noqa: E402
    entity_counts_per_sentence,
    evaluate_predictions,
    format_table,
    gold_tags,
    holm_adjust,
    paired_bootstrap,
)
from spanish_ner.modeling import predict_sentences  # noqa: E402
from spanish_ner.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="*",
        default=["models/bert-base-spanish-wwm-cased/best"],
        help="Paths to fine-tuned checkpoints to evaluate",
    )
    p.add_argument("--skip-baselines", action="store_true")
    return p.parse_args()


def score_both_views(
    predict_fn,
    test_sents,
    clean_sents,
    clean_index: list[int],
) -> dict:
    """Score a model on the full test set and on the deduplicated subset.

    The deduplicated score is computed by *slicing the same predictions*, not
    by re-running the model, so the two views cannot disagree for any reason
    other than the sentences included.
    """
    predictions = predict_fn(test_sents)
    full = evaluate_predictions(gold_tags(test_sents), predictions)
    dedup = evaluate_predictions(
        gold_tags(clean_sents), [predictions[i] for i in clean_index]
    )
    return {
        "full": full,
        "deduplicated": dedup,
        # Kept for the paired bootstrap: significance must be computed on the
        # same sentences for every model.
        "_counts": entity_counts_per_sentence(gold_tags(test_sents), predictions),
    }


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    splits = load_all(config, include_test=True)
    train, test = splits["train"], splits["test"]
    # Named separately because the two counts differ and both get reported:
    # `absent` is the 1,208 test sentences not present in training, and
    # `clean_keys` is already a set, so it is the 1,166 *distinct* ones.
    absent = deduplicate_against(test, train)
    clean_keys = {" ".join(s.tokens) for s in absent}

    # Index positions of the deduplicated sentences within the full test set.
    #
    # Two deduplications happen here and only the first used to be described.
    # `deduplicate_against` removes test sentences that occur in training;
    # `seen` then removes test sentences that occur more than once in the test
    # set itself. Both are defensible for a "generalisation to unseen text"
    # column -- boilerplate repeated four times should not be weighted four
    # times -- but the second was silent, and the prose reported the count from
    # between the two steps.
    seen: set[str] = set()
    clean_index = []
    for i, sent in enumerate(test):
        key = " ".join(sent.tokens)
        if key in clean_keys and key not in seen:
            seen.add(key)
            clean_index.append(i)
    clean = [test[i] for i in clean_index]

    log.info("Test set: %d sentences | %d absent from training | %d distinct "
             "and absent, which is what the deduplicated column scores",
             len(test), len(absent), len(clean))

    results: dict[str, dict] = {}

    if not args.skip_baselines:
        log.info("Evaluating gazetteer ...")
        gaz = GazetteerTagger().fit(train)
        results["gazetteer"] = score_both_views(gaz.predict, test, clean, clean_index)

        log.info("Evaluating CRF (retraining, ~50s) ...")
        crf = CRFTagger(**config["crf"]).fit(train)
        results["crf"] = score_both_views(crf.predict, test, clean, clean_index)

    max_length = config["transformer"]["max_length"]
    batch_size = config["transformer"]["eval_batch_size"]

    for model_path in args.models:
        path = PROJECT_ROOT / model_path
        if not path.exists():
            log.warning("Skipping missing checkpoint %s", path)
            continue
        name = path.parent.name
        log.info("Evaluating %s ...", name)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForTokenClassification.from_pretrained(path)
        results[name] = score_both_views(
            lambda s, m=model, t=tokenizer: predict_sentences(
                s, m, t, max_length, batch_size
            ),
            test, clean, clean_index,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- significance ----------------------------------------------------
    # Ranking models by a point estimate alone is not defensible: a one-point
    # F1 gap on 1,517 sentences is often within sampling noise.
    #
    # An earlier version of this block did something worse than not testing. It
    # sorted the models by their TEST F1 and then compared every other model
    # against the winner:
    #
    #     ranked = sorted(results, key=lambda n: results[n]["full"]["overall"]["f1"])
    #     for challenger in ranked[1:]: paired_bootstrap(counts[ranked[0]], ...)
    #
    # The reference arm was therefore chosen by its score on the very data the
    # p-values are computed from. Testing the maximum of a set against the rest,
    # on the sample that produced the maximum, biases every comparison toward
    # the reference -- and the four resulting p-values were printed as if each
    # were a single pre-registered test.
    counts = {name: res.pop("_counts") for name, res in results.items()}

    # The hypothesis this project was built to test, fixed before any score was
    # seen: does a Spanish-specific encoder beat a multilingual one on Spanish?
    # It is the confirmatory comparison and it is reported unadjusted.
    confirmatory = None
    if CONFIRMATORY[0] in counts and CONFIRMATORY[1] in counts:
        log.info("Confirmatory: %s vs %s ...", *CONFIRMATORY)
        confirmatory = {
            "comparison": f"{CONFIRMATORY[0]}_vs_{CONFIRMATORY[1]}",
            "prespecified": True,
            **paired_bootstrap(counts[CONFIRMATORY[0]], counts[CONFIRMATORY[1]]),
        }

    # Everything else is exploratory: all pairs, in a fixed alphabetical order
    # that owes nothing to the scores, with a Holm correction over the family.
    names = sorted(counts)
    exploratory = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            log.info("Bootstrapping %s vs %s ...", a, b)
            exploratory[f"{a}_vs_{b}"] = paired_bootstrap(counts[a], counts[b])

    adjusted = holm_adjust({k: v["p_value"] for k, v in exploratory.items()})
    for key, result in exploratory.items():
        result["p_value_holm"] = adjusted[key]
        result["significant_unadjusted"] = result.pop("significant", None)
        result["significant"] = bool(adjusted[key] < 0.05)

    save_json(
        {
            "split_sizes": {
                "test_sentences": len(test),
                "absent_from_training": len(absent),
                "distinct_and_absent_scored": len(clean),
                "note": ("The deduplicated column scores "
                         "distinct_and_absent_scored, not absent_from_training; "
                         "the two differ by the test sentences that repeat "
                         "within the test set."),
            },
            "per_model": results,
            "confirmatory": confirmatory,
            "significance": exploratory,
            "multiplicity_note": (
                f"{len(exploratory)} pairwise comparisons, Holm-adjusted. The "
                "reference arm is no longer chosen by test score; the one "
                "pre-specified comparison is reported separately and "
                "unadjusted."
            ),
            # Stored so the significance structure can be revisited without
            # reloading five models: these counts are what the bootstrap
            # resamples, and they were previously discarded.
            "entity_counts_per_sentence": {
                name: array.tolist() for name, array in counts.items()
            },
        },
        "reports/metrics_test.json",
    )

    print()
    print("| Model | Test F1 (full) | Test F1 (deduplicated) | Precision | Recall |")
    print("|---|---:|---:|---:|---:|")
    for name, res in results.items():
        f = res["full"]["overall"]
        d = res["deduplicated"]["overall"]
        print(f"| {name} | {f['f1']:.4f} | {d['f1']:.4f} | {f['precision']:.4f} | "
              f"{f['recall']:.4f} |")
    print()
    if confirmatory is not None:
        print("Pre-specified comparison (Spanish-specific vs multilingual), "
              "10,000 resamples, no multiplicity adjustment:")
        print()
        print("| Comparison | delta F1 | 95% CI | p | Significant |")
        print("|---|---:|---|---:|:--:|")
        print(f"| {confirmatory['comparison'].replace('_vs_', ' vs ')} | "
              f"{confirmatory['observed_difference']:+.4f} | "
              f"[{confirmatory['ci_lower']:+.4f}, {confirmatory['ci_upper']:+.4f}] | "
              f"{confirmatory['p_value']:.4f} | "
              f"{'yes' if confirmatory['significant'] else 'NO'} |")
        print()

    print(f"All {len(exploratory)} pairwise comparisons, exploratory, "
          f"Holm-adjusted:")
    print()
    print("| Comparison | delta F1 | 95% CI | p | p (Holm) | Significant |")
    print("|---|---:|---|---:|---:|:--:|")
    for key, c in sorted(exploratory.items(), key=lambda kv: kv[1]["p_value_holm"]):
        print(f"| {key.replace('_vs_', ' vs ')} | {c['observed_difference']:+.4f} | "
              f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}] | {c['p_value']:.4f} | "
              f"{c['p_value_holm']:.4f} | "
              f"{'yes' if c['significant'] else 'NO'} |")
    print()
    print("The reference arm is no longer the test-set winner, and a comparison "
          "that clears 0.05 only before adjustment is not a finding.")
    print()

    for name, res in results.items():
        print(format_table(res["full"], title=f"{name} -- test set (esp.testb, full)"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
