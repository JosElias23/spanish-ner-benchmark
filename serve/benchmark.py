"""Measure serving latency and throughput, and write the numbers to reports/.

"It runs fast" is not a claim anyone can check. This produces the ones they can:
p50, p95 and p99 latency, documents per second, and cost per million documents
at a stated hourly rate.

The benchmark drives the model directly rather than going over HTTP, so the
numbers isolate inference from the web stack. The HTTP overhead is small and
measuring it would say more about uvicorn than about the model.

Two things are deliberately measured separately:

**Batch size.** Transformer throughput is dominated by how many sequences share
a forward pass. Reporting single-document latency alone understates a real
deployment by an order of magnitude; reporting only batched throughput hides
what an interactive user actually waits for. Both are here.

**Warm-up.** The first forward pass pays for lazy CUDA context creation and
memory allocation. Including it in a p50 would be dishonest, so warm-up
iterations run first and are discarded.

Usage:
    python serve/benchmark.py
    python serve/benchmark.py --device cpu --iterations 5
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from spanish_ner.data import Sentence, load_split  # noqa: E402
from spanish_ner.modeling import predict_sentences  # noqa: E402
from spanish_ner.utils import PROJECT_ROOT, load_config, save_json, setup_logging  # noqa: E402

BATCH_SIZES = (1, 4, 8, 16, 32, 64)
WARMUP = 5

# Divisible by every batch size above, so every row of the table processes the
# identical workload and the only thing that changes is how it is grouped.
POOL_DOCUMENTS = 512

# Representative on-demand prices, September 2026. Stated explicitly because a
# cost figure without its assumption is not a cost figure.
# These are rented-hardware prices, and the throughput they are multiplied by is
# this machine's. The cost column is therefore "what this throughput would cost
# at this rate", not "what this workload costs on a T4" -- an RTX 5060 Ti is
# considerably faster than a T4, so a real T4 would bill more per million
# documents. The mismatch is named in the report and in the README rather than
# left for a reader to infer from a rate key.
HOURLY_RATES_USD = {
    "cpu_2vcpu": 0.10,   # a small general-purpose instance
    "gpu_t4": 0.53,      # NVIDIA T4, an entry-level inference GPU
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--out", default=None,
                   help="report path; defaults to reports/metrics_serving.json "
                        "for cuda and reports/metrics_serving_cpu.json for cpu, "
                        "so a CPU run cannot silently overwrite a GPU one")
    p.add_argument("--iterations", type=int, default=3,
                   help="passes over the 512-document pool per batch "
                        "size; each pass times every batch in it")
    p.add_argument("--model", default=None)
    return p.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model or str(
        PROJECT_ROOT / "models" / config["transformer"]["model_name"].split("/")[-1] / "best"
    )
    if not Path(model_path).exists():
        log.error("No checkpoint at %s. Run scripts/train_transformer.py first.", model_path)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()
    max_length = config["transformer"]["max_length"]

    # Real sentences from the development split, so token counts and therefore
    # latency reflect the distribution the model actually sees.
    #
    # Every batch size processes this SAME pool, in chunks. An earlier version
    # timed `pool[:batch_size]`, so batch 1 was timed on one specific sentence
    # and batch 64 on a different set of 64 -- which makes the cross-row
    # comparison, and the headline batching speed-up read off it, a mixture of
    # batch size and which documents happened to be measured. 512 is divisible
    # by every batch size here, so each row does identical work.
    dev = load_split("dev", config)
    pool: list[Sentence] = dev[:POOL_DOCUMENTS]
    mean_tokens = float(np.mean([len(s) for s in pool]))
    log.info("Device %s | %d parameters | %d documents, mean %.1f tokens each",
             device, sum(p.numel() for p in model.parameters()), len(pool),
             mean_tokens)

    results = []
    for batch_size in BATCH_SIZES:
        chunks = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
        chunks = [c for c in chunks if len(c) == batch_size]
        documents = len(chunks) * batch_size

        for _ in range(WARMUP):
            predict_sentences(chunks[0], model, tokenizer, max_length, batch_size,
                              device)
        if device == "cuda":
            torch.cuda.synchronize()

        # Two quantities, deliberately separated. `latencies_ms` is what one
        # caller waits for one batch, now measured over many different batches
        # rather than one. `pass_seconds` is the wall clock for the identical
        # 512-document workload, which is what the throughput column is about.
        latencies_ms: list[float] = []
        pass_seconds: list[float] = []
        for _ in range(args.iterations):
            pass_started = time.perf_counter()
            for chunk in chunks:
                started = time.perf_counter()
                predict_sentences(chunk, model, tokenizer, max_length, batch_size,
                                  device)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - started) * 1000)
            pass_seconds.append(time.perf_counter() - pass_started)

        p50 = percentile(latencies_ms, 50)
        throughput = documents / float(np.median(pass_seconds))
        rate = HOURLY_RATES_USD["gpu_t4" if device == "cuda" else "cpu_2vcpu"]

        entry = {
            "batch_size": batch_size,
            "documents_timed": documents,
            "batches_timed": len(chunks) * args.iterations,
            "p50_ms": round(p50, 2),
            "p95_ms": round(percentile(latencies_ms, 95), 2),
            "p99_ms": round(percentile(latencies_ms, 99), 2),
            "mean_ms": round(float(np.mean(latencies_ms)), 2),
            "ms_per_document": round(p50 / batch_size, 3),
            "documents_per_second": round(throughput, 1),
            "usd_per_million_documents": round(1e6 / throughput / 3600 * rate, 2),
        }
        results.append(entry)
        log.info(
            "batch %2d | p50 %7.2f ms | p95 %7.2f ms | %7.1f docs/s | "
            "USD %6.2f per million docs",
            batch_size, entry["p50_ms"], entry["p95_ms"],
            entry["documents_per_second"], entry["usd_per_million_documents"],
        )

    best = max(results, key=lambda r: r["documents_per_second"])
    single = results[0]
    speedup = best["documents_per_second"] / single["documents_per_second"]

    # Store the checkpoint location relative to the repository root. An absolute
    # path would publish the developer's home directory and folder layout into a
    # committed artefact, which is both a small information leak and useless to
    # anyone else reading the file.
    try:
        recorded_model = str(Path(model_path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        recorded_model = Path(model_path).name

    save_json(
        {
            "environment": {
                "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "torch": torch.__version__,
                "platform": f"{platform.system()} {platform.release()}",
                "python": platform.python_version(),
            },
            "model": recorded_model,
            "max_length": max_length,
            "passes_per_batch_size": args.iterations,
            "warmup_iterations": WARMUP,
            "documents_per_row": POOL_DOCUMENTS,
            "mean_tokens_per_document": round(mean_tokens, 1),
            "hourly_rate_usd": HOURLY_RATES_USD,
            "priced_hardware": "gpu_t4" if device == "cuda" else "cpu_2vcpu",
            "measured_hardware": (
                torch.cuda.get_device_name(0) if device == "cuda" else "this CPU"),
            "cost_note": (
                "Cost assumes sustained utilisation at the stated on-demand hourly "
                "rate and excludes network, storage and orchestration overhead. The "
                "rate and the throughput come from DIFFERENT hardware: the price is "
                "an NVIDIA T4's and the throughput is this machine's, so on a real "
                "T4 the cost per million documents would be higher. The column is a "
                "rate times a measured throughput, not a quote."
            ),
            "workload_note": (
                "Every batch size processes the same "
                f"{POOL_DOCUMENTS} documents, in chunks. p50/p95/p99 are per-batch "
                "latencies over every chunk of every pass; documents_per_second is "
                "the whole pool divided by the median pass time."
            ),
            "by_batch_size": results,
            "batching_speedup": round(speedup, 1),
        },
        args.out or ("reports/metrics_serving.json" if device == "cuda"
                     else "reports/metrics_serving_cpu.json"),
    )

    print()
    print("| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Docs/s | USD / 1M docs |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['batch_size']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | "
              f"{r['p99_ms']:.1f} | {r['documents_per_second']:.0f} | "
              f"{r['usd_per_million_documents']:.2f} |")
    print()
    print(f"Batching {best['batch_size']} documents gives {speedup:.1f}x the throughput "
          f"of one-at-a-time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
