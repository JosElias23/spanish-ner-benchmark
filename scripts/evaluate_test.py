"""Final evaluation on the held-out test split (esp.testb).

This is the only script in the repository that loads the test set. Everything
else -- baseline selection, hyperparameters, early stopping, error analysis --
happens on the development split. Running this script is the last step, and its
output is what the README reports.

Two numbers are produced for every model:

  full        the official 1,517-sentence test set, comparable with published
              CoNLL-2002 results
  deduplicated the 1,208 test sentences that do not appear verbatim in
              training, a stricter estimate of generalisation to unseen text

Usage:
    python scripts/evaluate_test.py --models models/bert-base-spanish-wwm-cased/best
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from spanish_ner.baselines import CRFTagger, GazetteerTagger  # noqa: E402
from spanish_ner.data import deduplicate_against, load_all  # noqa: E402
from spanish_ner.evaluate import (  # noqa: E402
    entity_counts_per_sentence,
    evaluate_predictions,
    format_table,
    gold_tags,
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
    clean = deduplicate_against(test, train)
    clean_keys = {" ".join(s.tokens) for s in clean}

    # Index positions of the deduplicated sentences within the full test set.
    seen: set[str] = set()
    clean_index = []
    for i, sent in enumerate(test):
        key = " ".join(sent.tokens)
        if key in clean_keys and key not in seen:
            seen.add(key)
            clean_index.append(i)
    clean = [test[i] for i in clean_index]

    log.info("Test set: %d sentences (%d after removing train duplicates)",
             len(test), len(clean))

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
    counts = {name: res.pop("_counts") for name, res in results.items()}
    ranked = sorted(results, key=lambda n: results[n]["full"]["overall"]["f1"], reverse=True)
    comparisons = {}
    for challenger in ranked[1:]:
        key = f"{ranked[0]}_vs_{challenger}"
        log.info("Bootstrapping %s ...", key)
        comparisons[key] = paired_bootstrap(counts[ranked[0]], counts[challenger])

    save_json(
        {"per_model": results, "significance": comparisons},
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
    print(f"Paired bootstrap vs best model ({ranked[0]}), 10,000 resamples:")
    print()
    print("| Comparison | delta F1 | 95% CI | p | Significant |")
    print("|---|---:|---|---:|:--:|")
    for key, c in comparisons.items():
        print(f"| {key.replace('_vs_', ' vs ')} | {c['observed_difference']:+.4f} | "
              f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}] | {c['p_value']:.4f} | "
              f"{'yes' if c['significant'] else 'NO'} |")
    print()
    for name, res in results.items():
        print(format_table(res["full"], title=f"{name} -- test set (esp.testb, full)"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
