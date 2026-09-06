"""FastAPI service for Spanish NER.

The Gradio demo shows the model works. This shows it can be operated: a typed
HTTP contract, batching, health and readiness probes, per-request latency
headers, and a Prometheus-style metrics endpoint.

Design decisions worth defending:

**One model, loaded once, at startup.** Loading per request would dominate
latency and make the p99 meaningless. The readiness probe stays false until the
weights are in memory, so an orchestrator does not route traffic to a pod that
cannot serve it.

**Batching is exposed in the API, not hidden.** Transformer throughput is
dominated by how many sequences share a forward pass. A caller that sends 32
documents in one request gets several times the throughput of 32 separate
calls, and the benchmark in `serve/benchmark.py` quantifies exactly that.

**Inference runs through `predict_sentences`**, the same function that produced
every metric in the README. A service that disagrees with its own evaluation is
worse than no service.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

from spanish_ner.data import Sentence, iter_entities  # noqa: E402
from spanish_ner.modeling import predict_sentences  # noqa: E402

MODEL_PATH = os.environ.get(
    "NER_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "models" /
        "bert-base-spanish-wwm-cased" / "best"),
)
MAX_LENGTH = int(os.environ.get("NER_MAX_LENGTH", "256"))
MAX_BATCH = int(os.environ.get("NER_MAX_BATCH", "64"))

# Matches the corpus tokenisation closely enough for free text.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

STATE: dict = {"ready": False, "model": None, "tokenizer": None, "device": "cpu"}
# Bounded so a long-running process cannot grow without limit.
LATENCIES: deque[float] = deque(maxlen=10_000)
COUNTERS = {"requests": 0, "documents": 0, "entities": 0, "errors": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device).eval()
    STATE.update(ready=True, model=model, tokenizer=tokenizer, device=device)
    yield
    STATE.update(ready=False, model=None, tokenizer=None)


app = FastAPI(
    title="Spanish NER",
    description="Named entity recognition for Spanish, fine-tuned on CoNLL-2002.",
    version="0.1.0",
    lifespan=lifespan,
)


class ExtractRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH,
        description="Spanish documents. Batch them: throughput scales with batch size.",
    )


class Entity(BaseModel):
    text: str
    type: str
    start_token: int
    end_token: int


class Document(BaseModel):
    text: str
    entities: list[Entity]


class ExtractResponse(BaseModel):
    documents: list[Document]
    model: str
    latency_ms: float
    device: str


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health", response_model=None)
def health() -> dict:
    """Liveness: the process is up. Always cheap, never touches the model."""
    return {"status": "ok"}


@app.get("/ready", response_model=None)
def ready() -> dict:
    """Readiness: the weights are loaded and the service can actually answer.

    Separate from /health on purpose. Model loading takes seconds, and an
    orchestrator that routes traffic on liveness alone will send requests to a
    pod that is still warming up.
    """
    return {"ready": STATE["ready"], "device": STATE["device"], "model": MODEL_PATH}


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    started = time.perf_counter()

    sentences = []
    token_lists = []
    for text in request.texts:
        tokens = TOKEN_PATTERN.findall(text) or ["."]
        token_lists.append(tokens)
        sentences.append(
            Sentence(tokens=tokens, pos_tags=["X"] * len(tokens),
                     ner_tags=["O"] * len(tokens))
        )

    tag_lists = predict_sentences(
        sentences, STATE["model"], STATE["tokenizer"], MAX_LENGTH,
        batch_size=len(sentences),
    )

    documents = []
    for text, tokens, tags in zip(request.texts, token_lists, tag_lists, strict=True):
        entities = [
            Entity(
                text=" ".join(tokens[start:end]),
                type=entity_type,
                start_token=start,
                end_token=end,
            )
            for entity_type, start, end in iter_entities(tags)
        ]
        COUNTERS["entities"] += len(entities)
        documents.append(Document(text=text, entities=entities))

    elapsed_ms = (time.perf_counter() - started) * 1000
    LATENCIES.append(elapsed_ms)
    COUNTERS["requests"] += 1
    COUNTERS["documents"] += len(request.texts)

    return ExtractResponse(
        documents=documents,
        model=Path(MODEL_PATH).parent.name,
        latency_ms=round(elapsed_ms, 2),
        device=STATE["device"],
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text exposition of counters and latency quantiles.

    Quantiles are computed over the last 10,000 requests rather than since
    start-up: an operator cares what the service is doing now, not what it did
    once during a cold start three days ago.
    """
    lines = [
        "# HELP ner_requests_total Extraction requests served.",
        "# TYPE ner_requests_total counter",
        f"ner_requests_total {COUNTERS['requests']}",
        "# HELP ner_documents_total Documents processed.",
        "# TYPE ner_documents_total counter",
        f"ner_documents_total {COUNTERS['documents']}",
        "# HELP ner_entities_total Entities returned.",
        "# TYPE ner_entities_total counter",
        f"ner_entities_total {COUNTERS['entities']}",
    ]

    if LATENCIES:
        ordered = sorted(LATENCIES)

        def quantile(q: float) -> float:
            index = min(len(ordered) - 1, int(q * len(ordered)))
            return ordered[index]

        lines += [
            "# HELP ner_request_latency_ms Request latency over the last 10k requests.",
            "# TYPE ner_request_latency_ms summary",
            f'ner_request_latency_ms{{quantile="0.5"}} {quantile(0.50):.2f}',
            f'ner_request_latency_ms{{quantile="0.95"}} {quantile(0.95):.2f}',
            f'ner_request_latency_ms{{quantile="0.99"}} {quantile(0.99):.2f}',
        ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
