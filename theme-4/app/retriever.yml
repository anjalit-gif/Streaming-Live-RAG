import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    # Regex tokenizer instead of naive .split(" ") - strips punctuation so
    # "hours." and "hours" match for BM25 instead of being different tokens.
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(self, documents: List[Dict[str, Any]], embed_model: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.corpus_texts = [doc["text"] for doc in documents]

        t0 = time.time()

        tokenized_corpus = [_tokenize(doc) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.embedder = SentenceTransformer(embed_model)
        self.embeddings = self.embedder.encode(
            self.corpus_texts, convert_to_numpy=True, normalize_embeddings=True
        )

        # Tracked so the pipeline can report "indexing cost" as the spec asks for.
        self.index_time_s = round(time.time() - t0, 3)

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec  # both normalized -> dot product = cosine similarity
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def hybrid_retrieve(self, query: str, top_k: int = 3, rrf_k: int = None) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion combining sparse (BM25) and dense (embedding) rankings."""
        rrf_k = rrf_k or settings.RRF_K

        dense_results = self._dense_search(query, top_k=top_k * 2)
        sparse_results = self._sparse_search(query, top_k=top_k * 2)
        dense_lookup = dict(dense_results)

        rrf_scores: Dict[int, float] = {}
        for rank, (doc_idx, _) in enumerate(dense_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (doc_idx, _) in enumerate(sparse_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        retrieved = []
        for doc_idx, rrf_score in sorted_docs:
            doc_data = self.documents[doc_idx].copy()
            # Cosine similarity (not the RRF score) is what the uncertainty gate checks,
            # since RRF scores aren't on a comparable scale across queries.
            doc_data["score"] = dense_lookup.get(doc_idx, 0.0)
            doc_data["rrf_score"] = rrf_score
            retrieved.append(doc_data)
        return retrieved
