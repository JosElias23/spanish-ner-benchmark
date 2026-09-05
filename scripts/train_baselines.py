"""Train the gazetteer and CRF baselines and report dev-set metrics.

Usage:
    python scripts/train_baselines.py

Writes reports/metrics_baselines_dev.json. The test split is deliberately
untouched here: baselines are selected on dev, exactly like the transformer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spanish_ner.baselines import CRFTagger, GazetteerTagger  # noqa: E402
from spanish_ner.data import load_all  # noqa: E402
from spanish_ner.evaluate import evaluate_predictions, format_table, gold_tags  # noqa: E402
from spanish_ner.utils import load_config, save_json, set_seed, setup_logging  # noqa: E402


def main() -> int:
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    splits = load_all(config)
    train, dev = splits["train"], splits["dev"]
    log.info("Loaded %d train / %d dev sentences", len(train), len(dev))

    y_true = gold_tags(dev)
    results = {}

    # --- Baseline 1: gazetteer -------------------------------------------
    t0 = time.perf_counter()
    gaz = GazetteerTagger().fit(train)
    fit_s = time.perf_counter() - t0
    log.info("Gazetteer: memorised %d distinct surface forms in %.2fs",
             len(gaz.gazetteer), fit_s)
    res = evaluate_predictions(y_true, gaz.predict(dev))
    res["train_seconds"] = round(fit_s, 2)
    results["gazetteer"] = res
    log.info("Gazetteer dev F1 = %.4f", res["overall"]["f1"])

    # --- Baseline 2: CRF --------------------------------------------------
    crf_cfg = config["crf"]
    t0 = time.perf_counter()
    crf = CRFTagger(**crf_cfg).fit(train)
    fit_s = time.perf_counter() - t0
    log.info("CRF trained in %.1fs", fit_s)
    res = evaluate_predictions(y_true, crf.predict(dev))
    res["train_seconds"] = round(fit_s, 2)
    results["crf"] = res
    log.info("CRF dev F1 = %.4f", res["overall"]["f1"])

    save_json(results, "reports/metrics_baselines_dev.json")

    print()
    for name, res in results.items():
        print(format_table(res, title=f"{name} -- development set (esp.testa)"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
