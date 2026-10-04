import time
from typing import Any, Dict

from app.pipeline import LiveRAGPipeline
from app.retriever import HybridRetriever

CORPUS = [
    {"id": "Doc_A", "text": "Standard travel allowance is $50 per day for domestic trips."},
    {"id": "Doc_B", "text": "Conference tickets are fully non-refundable once issued."},
    {"id": "Doc_C", "text": "Hotel bookings over 3 nights require manager pre-approval."},
]

TEST_CASES = [
    {
        "name": "Out-of-corpus query -> honest 'I don't know'",
        "query": "What is the orbital velocity of Mars?",
        "check": lambda r: r.status == "uncertain" and r.uncertainty is not None,
    },
    {
        "name": "Multi-intent decomposition",
        "query": "What is the travel allowance, and are conference tickets refundable?",
        "check": lambda r: len(r.sub_queries) >= 2,
    },
    {
        "name": "Simple single-fact query",
        "query": "What is the daily travel allowance amount?",
        "check": lambda r: r.status == "answered" and len(r.citations) >= 1,
    },
    {
        "name": "Too-short utterance -> WAIT, not answered",
        "query": "travel cost",
        "check": lambda r: r.status == "waiting",
    },
    {
        "name": "Query suppression gate (reformat request)",
        "setup_answer_first": True,
        "query": "summarize in 2 sentences",
        "check": lambda r: r.status == "no_retrieval" and len(r.retrieval_events) == 0,
    },
]


def run_adversarial_benchmark() -> Dict[str, Any]:
    # FIX: each test case now gets its own fresh retriever + pipeline instance.
    # Previously all cases shared one pipeline, so test 5 only passed because
    # earlier tests had already populated session state - a hidden order
    # dependency that would have given misleading/irreproducible results.
    print("=== RUNNING ADVERSARIAL BENCHMARK SUITE ===")
    results = []

    for idx, tc in enumerate(TEST_CASES, start=1):
        retriever = HybridRetriever(CORPUS)
        pipeline = LiveRAGPipeline(retriever)

        if tc.get("setup_answer_first"):
            pipeline.process_query("What is the hotel approval policy?")

        t0 = time.time()
        result = pipeline.process_query(tc["query"])
        latency_ms = round((time.time() - t0) * 1000, 2)

        passed = bool(tc["check"](result))
        status_label = "PASSED" if passed else "FAILED"
        print(f"[{status_label}] Test {idx}: {tc['name']} | latency={latency_ms}ms | status={result.status}")
        results.append({"name": tc["name"], "passed": passed, "latency_ms": latency_ms})

    pass_count = sum(r["passed"] for r in results)
    avg_latency = round(sum(r["latency_ms"] for r in results) / len(results), 2)
    print(f"\n{pass_count}/{len(results)} tests passed | avg latency {avg_latency}ms")
    return {"results": results, "pass_count": pass_count, "total": len(results), "avg_latency_ms": avg_latency}


if __name__ == "__main__":
    run_adversarial_benchmark()
