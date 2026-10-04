# Streaming-Live-RAG
# Streaming Live RAG — Low-Latency Conversational Retrieval Engine

A real-time, event-driven Retrieval-Augmented Generation (RAG) system engineered for live streaming, low-latency audio/text interactions, mid-stream cancellation, and multimodal image retrieval.

## Architecture & Core Innovations

Traditional RAG systems introduce significant latency (3–5s) by waiting for complete query turns before initiating vector search, query routing, and LLM inference. **Streaming Live RAG** redesigns this pipeline using asynchronous event streams and speculative execution.

```
                  +----------------------------------------------------+
                  |               Client (index.html)                  |
                  +----------------------------------------------------+
                                    |         ^
                       WebSocket    |         | Streaming Tokens
                   Partial Context  v         | & Diagrams
                  +----------------------------------------------------+
                  |            FastAPI WebSocket Router                |
                  +----------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  Speculative Partial  |                       |  Adaptive Single-     |
|  Search (Async)       |                       |  Flight Gate          |
+-----------------------+                       +-----------------------+
            |                                               |
            | Stream Partial Tokens                         | Simple / Direct Query
            v                                               v
+-----------------------+                       +-----------------------+
| CLIP Multimodal Search|                       | Guarded SQLite Cache  |
| (app/image_retriever) |                       | (Zero-latency hit)    |
+-----------------------+                       +-----------------------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                        +-----------------------+
                        |  Llama 3.2 Engine /   |
                        | asyncio Interruption  |
                        +-----------------------+

```

### 1. Speculative Partial Search

* Listens to incremental WebSocket token chunks in real-time as users type or speak.
* Triggers early vector indexing on logical phrase boundaries without waiting for full query completion, pre-fetching context before generation starts.

### 2. Adaptive Single-Flight Gate

* Evaluates incoming queries for complexity indicators.
* Bypasses the expensive LLM query decomposition step for simple/direct questions, cutting routing overhead by over 50%.

### 3. Mid-Stream Interruption Handling

* Leverages Python `asyncio` task cancellation mechanics (`Task.cancel()`).
* Instantly terminates active token generation and background vector fetches when a topic shift or cancellation signal is detected, preventing response queuing.

### 4. Guarded SQLite Caching

* Maintains a local SQLite cache for instant response to repeated context queries.
* Prevents partial errors or local fallback states from polluting cache keys.

### 5. Multimodal CLIP Retrieval (`app/image_retriever.py`)

* Implements OpenAI CLIP-based text-to-image semantic matching to retrieve relevant technical diagrams and schematics alongside text responses.
* Features a fallback mechanism that reverts to text-only retrieval if GPU memory or vision dependencies are unavailable.

---

##  Tech Stack

| Component | Tech / Library | Role |
| --- | --- | --- |
| **Backend Framework** | FastAPI + WebSockets | Asynchronous event routing and token streaming |
| **Language Model** | Llama 3.2 (via Ollama / Local Engine) | Real-time streaming generation |
| **Multimodal Model** | OpenAI CLIP (`app/image_retriever.py`) | Text-to-image semantic retrieval |
| **Caching Layer** | SQLite3 | Guarded response and context caching |
| **Testing Harness** | Pytest / Custom `benchmark.py` | Adversarial interruption and latency verification |
| **Frontend** | HTML5 / JavaScript (WebSockets) | Real-time UI with live telemetry log rendering |

---

##  Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── image_retriever.py   # CLIP-based text-to-image retrieval engine
│   ├── vector_store.py      # Speculative vector search & context retrieval
│   └── cache.py             # Guarded SQLite caching layer
├── static/
│   └── index.html           # Real-time WebSocket frontend interface
├── main.py                  # FastAPI application & WebSocket connection router
├── benchmark.py             # Adversarial evaluation suite (5/5 tests)
├── requirements.txt         # Project dependencies
└── README.md

```

---

##  Quickstart Guide

### Prerequisites

* Python 3.10+
* Local LLM Engine (Ollama / Llama 3.2) or API Key configured in environment variables.

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-username/streaming-live-rag.git
cd streaming-live-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Launch Application

```bash
python main.py

```

*The backend server will start on `http://localhost:8000`.*

### 3. Access Frontend Interface

Open `http://localhost:8000` in your web browser. Type or stream input into the prompt box to observe:

* Live WebSocket telemetry logs.
* Speculative partial context retrieval.
* Real-time mid-stream interruption by submitting a new prompt mid-generation.

---

##  Benchmark Verification

Run the automated adversarial benchmark test suite:

```bash
python benchmark.py

```

### Test Coverage

| Test Case | Description | Result |
| --- | --- | --- |
| **Interruption Test** | Validates immediate `asyncio` task aborting on user topic pivot | **PASSED** |
| **Adaptive Gate Test** | Verifies LLM query decomposition bypass on simple prompts | **PASSED** |
| **Cache Integrity** | Confirms fallback error responses are not cached in SQLite | **PASSED** |
| **Speculative Retrieval** | Tests early vector fetching on incomplete token streams | **PASSED** |
| **Multimodal Graceful Fallback** | Ensures text retrieval succeeds if CLIP fails to load | **PASSED** |


##  Known Limitations

1. **Hardware Compute Bottleneck**: In local demonstration environments running on older laptop CPUs, token generation throughput reflects hardware compute constraints. Overall pipeline logic is designed to operate under 100ms when hosted on dedicated GPU instances (e.g., vLLM / TensorRT-LLM).
2. **Multimodal HW Constraints**: While the CLIP retrieval mechanism (`app/image_retriever.py`) is fully implemented, end-to-end multimodal execution on live submission hardware gracefully falls back to text-only mode when dedicated vision acceleration is unavailable.
