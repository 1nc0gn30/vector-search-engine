"""Pure Python hybrid search engine combining dense vector search and sparse BM25 keyword search.

Implements Okapi BM25 inverted index tokenization and ranking, Reciprocal
Rank Fusion (RRF), and normalized linear alpha score blending.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from vector_search_engine.models import (
    DistanceMetric,
    HybridSearchResult,
    SearchResult,
    VectorItem,
)
from vector_search_engine.metrics import compute_similarity
from vector_search_engine.indexes import evaluate_filter


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """Pure Python Okapi BM25 sparse inverted index."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or simple_tokenize

        # Inverted index: term -> {doc_id: term_frequency}
        self._inverted_index: Dict[str, Dict[str, int]] = {}
        # Document length: doc_id -> token_count
        self._doc_lengths: Dict[str, int] = {}
        # Document raw text and metadata
        self._documents: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._total_tokens: int = 0

    def size(self) -> int:
        """Return total indexed documents."""
        return len(self._doc_lengths)

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index or update a document in the BM25 index."""
        if doc_id in self._doc_lengths:
            self.delete_document(doc_id)

        tokens = self.tokenizer(text or "")
        token_count = len(tokens)
        self._doc_lengths[doc_id] = token_count
        self._documents[doc_id] = text or ""
        self._metadata[doc_id] = dict(metadata or {})
        self._total_tokens += token_count

        # Term frequency counts
        tf_map: Dict[str, int] = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        for term, count in tf_map.items():
            if term not in self._inverted_index:
                self._inverted_index[term] = {}
            self._inverted_index[term][doc_id] = count

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the BM25 index."""
        if doc_id not in self._doc_lengths:
            return False

        old_len = self._doc_lengths.pop(doc_id)
        self._total_tokens -= old_len
        self._documents.pop(doc_id, None)
        self._metadata.pop(doc_id, None)

        # Remove from inverted index
        for term, postings in list(self._inverted_index.items()):
            if doc_id in postings:
                del postings[doc_id]
                if not postings:
                    del self._inverted_index[term]

        return True

    def clear(self) -> None:
        """Clear the entire BM25 index."""
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._documents.clear()
        self._metadata.clear()
        self._total_tokens = 0

    def search(
        self,
        query: str,
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search documents using Okapi BM25 scoring."""
        if k <= 0 or not self._doc_lengths:
            return []

        query_tokens = self.tokenizer(query or "")
        if not query_tokens:
            return []

        num_docs = len(self._doc_lengths)
        avg_doc_len = self._total_tokens / max(num_docs, 1)

        doc_scores: Dict[str, float] = {}

        for term in query_tokens:
            postings = self._inverted_index.get(term)
            if not postings:
                continue

            # Document frequency
            doc_freq = len(postings)
            # Okapi BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
            idf = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            if idf <= 0:
                idf = 1e-6

            for doc_id, tf in postings.items():
                meta = self._metadata.get(doc_id, {})
                if filter_expr and not evaluate_filter(meta, filter_expr):
                    continue

                d_len = self._doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(avg_doc_len, 1e-6)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-6))

                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

        # Convert scores to SearchResult objects
        results: List[SearchResult] = []
        for doc_id, score in doc_scores.items():
            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    distance=1.0 / (1.0 + score),  # synthetic distance
                    vector=None,
                    metadata=self._metadata.get(doc_id, {}),
                    document=self._documents.get(doc_id),
                )
            )

        results.sort(key=lambda r: -r.score)
        return results[:k]


class HybridSearchEngine:
    """Orchestrates dense vector search and BM25 sparse text search."""

    @staticmethod
    def fuse_rrf(
        dense_results: Sequence[SearchResult],
        sparse_results: Sequence[SearchResult],
        k: int = 60,
        top_k: int = 10,
    ) -> List[HybridSearchResult]:
        """Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

        RRF(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        combined_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}
        dense_scores_map: Dict[str, float] = {}
        sparse_scores_map: Dict[str, float] = {}
        metadata_map: Dict[str, Dict[str, Any]] = {}
        document_map: Dict[str, Optional[str]] = {}
        vector_map: Dict[str, Optional[List[float]]] = {}

        # Process dense results
        for rank, res in enumerate(dense_results, start=1):
            dense_ranks[res.id] = rank
            dense_scores_map[res.id] = res.score
            metadata_map[res.id] = res.metadata
            document_map[res.id] = res.document
            if res.vector is not None:
                vector_map[res.id] = res.vector
            combined_scores[res.id] = combined_scores.get(res.id, 0.0) + (1.0 / (k + rank))

        # Process sparse results
        for rank, res in enumerate(sparse_results, start=1):
            sparse_ranks[res.id] = rank
            sparse_scores_map[res.id] = res.score
            if res.id not in metadata_map:
                metadata_map[res.id] = res.metadata
            if res.id not in document_map:
                document_map[res.id] = res.document
            if res.vector is not None:
                vector_map[res.id] = res.vector
            combined_scores[res.id] = combined_scores.get(res.id, 0.0) + (1.0 / (k + rank))

        # Default rank if not present in one ranking
        missing_rank = max(len(dense_results), len(sparse_results)) + 1

        hybrid_results: List[HybridSearchResult] = []
        for doc_id, c_score in combined_scores.items():
            hybrid_results.append(
                HybridSearchResult(
                    id=doc_id,
                    combined_score=c_score,
                    dense_score=dense_scores_map.get(doc_id, 0.0),
                    sparse_score=sparse_scores_map.get(doc_id, 0.0),
                    dense_rank=dense_ranks.get(doc_id, missing_rank),
                    sparse_rank=sparse_ranks.get(doc_id, missing_rank),
                    metadata=metadata_map.get(doc_id, {}),
                    document=document_map.get(doc_id),
                    vector=vector_map.get(doc_id),
                )
            )

        hybrid_results.sort(key=lambda r: -r.combined_score)
        return hybrid_results[:top_k]

    @staticmethod
    def fuse_linear(
        dense_results: Sequence[SearchResult],
        sparse_results: Sequence[SearchResult],
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> List[HybridSearchResult]:
        """Combine dense and sparse search scores via Min-Max normalized linear blending.

        combined_score = alpha * norm_dense_score + (1 - alpha) * norm_sparse_score
        """
        alpha = max(0.0, min(1.0, float(alpha)))

        all_ids: Set[str] = {r.id for r in dense_results} | {r.id for r in sparse_results}
        if not all_ids:
            return []

        # Extract scores
        dense_scores = {r.id: r.score for r in dense_results}
        sparse_scores = {r.id: r.score for r in sparse_results}

        dense_ranks = {r.id: i for i, r in enumerate(dense_results, start=1)}
        sparse_ranks = {r.id: i for i, r in enumerate(sparse_results, start=1)}

        metadata_map = {r.id: r.metadata for r in dense_results}
        metadata_map.update({r.id: r.metadata for r in sparse_results})

        doc_map = {r.id: r.document for r in dense_results}
        doc_map.update({r.id: r.document for r in sparse_results})

        vector_map: Dict[str, Optional[List[float]]] = {}
        for r in dense_results:
            if r.vector is not None:
                vector_map[r.id] = r.vector

        # Min-Max Normalization helper
        def _normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
            if not score_dict:
                return {}
            vals = list(score_dict.values())
            min_v = min(vals)
            max_v = max(vals)
            span = max_v - min_v
            if span <= 1e-12:
                return {k: 1.0 for k in score_dict}
            return {k: (v - min_v) / span for k, v in score_dict.items()}

        norm_dense = _normalize_scores(dense_scores)
        norm_sparse = _normalize_scores(sparse_scores)

        missing_rank = max(len(dense_results), len(sparse_results)) + 1
        hybrid_results: List[HybridSearchResult] = []

        for doc_id in all_ids:
            nd = norm_dense.get(doc_id, 0.0)
            ns = norm_sparse.get(doc_id, 0.0)
            combined = alpha * nd + (1.0 - alpha) * ns

            hybrid_results.append(
                HybridSearchResult(
                    id=doc_id,
                    combined_score=combined,
                    dense_score=dense_scores.get(doc_id, 0.0),
                    sparse_score=sparse_scores.get(doc_id, 0.0),
                    dense_rank=dense_ranks.get(doc_id, missing_rank),
                    sparse_rank=sparse_ranks.get(doc_id, missing_rank),
                    metadata=metadata_map.get(doc_id, {}),
                    document=doc_map.get(doc_id),
                    vector=vector_map.get(doc_id),
                )
            )

        hybrid_results.sort(key=lambda r: -r.combined_score)
        return hybrid_results[:top_k]

    @staticmethod
    def maximal_marginal_relevance(
        query_vector: Sequence[float],
        candidates: Sequence[Union[SearchResult, VectorItem]],
        k: int = 10,
        lambda_mult: float = 0.5,
        metric: Union[DistanceMetric, str] = DistanceMetric.COSINE,
    ) -> List[SearchResult]:
        """Rank and select diverse candidates using Maximal Marginal Relevance (MMR).

        MMR balances query relevance with novelty/diversity by penalizing candidates
        that are highly similar to already selected candidates:
            MMR(d) = lambda * sim(d, query) - (1 - lambda) * max_{s in selected} sim(d, s)

        Args:
            query_vector: Search query embedding.
            candidates: Candidate search results or VectorItems (must include vector).
            k: Maximum number of diverse results to return.
            lambda_mult: Diversity weight in [0.0, 1.0].
                1.0 = Pure relevance (equivalent to nearest neighbors).
                0.0 = Maximal diversity / novelty.
                0.5 = Balanced relevance and diversity.
            metric: Distance metric for similarity evaluation.

        Returns:
            List of SearchResult items ranked by MMR selection order.
        """
        if not candidates or k <= 0:
            return []

        lambda_mult = max(0.0, min(1.0, float(lambda_mult)))
        metric_enum = DistanceMetric.from_str(metric) if isinstance(metric, str) else metric

        # Normalize candidates to (id, vector, SearchResult)
        item_pool: List[Tuple[str, Sequence[float], SearchResult]] = []
        for cand in candidates:
            if isinstance(cand, VectorItem):
                sim = compute_similarity(cand.vector, query_vector, metric_enum)
                dist = 1.0 - sim if metric_enum == DistanceMetric.COSINE else max(0.0, -sim)
                sr = SearchResult(
                    id=cand.id,
                    score=sim,
                    distance=dist,
                    vector=list(cand.vector),
                    metadata=dict(cand.metadata),
                    document=cand.document,
                )
                item_pool.append((cand.id, cand.vector, sr))
            elif isinstance(cand, SearchResult):
                if cand.vector is None:
                    continue  # Vector required for MMR diversity computation
                item_pool.append((cand.id, cand.vector, cand))

        if not item_pool:
            return []

        # Precompute query relevance similarity for all candidates
        query_sims: Dict[str, float] = {}
        for doc_id, vec, sr in item_pool:
            query_sims[doc_id] = compute_similarity(vec, query_vector, metric_enum)

        selected: List[SearchResult] = []
        selected_vectors: List[Sequence[float]] = []
        unselected = list(item_pool)

        # Iteratively select candidates maximizing MMR
        target_count = min(k, len(item_pool))
        while len(selected) < target_count and unselected:
            best_idx = -1
            best_mmr_score = -float("inf")

            for idx, (doc_id, vec, sr) in enumerate(unselected):
                sim_to_query = query_sims[doc_id]
                if not selected_vectors:
                    mmr_score = sim_to_query
                else:
                    max_sim_to_selected = max(
                        compute_similarity(vec, s_vec, metric_enum)
                        for s_vec in selected_vectors
                    )
                    mmr_score = lambda_mult * sim_to_query - (1.0 - lambda_mult) * max_sim_to_selected

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                doc_id, vec, sr = unselected.pop(best_idx)
                # Create result copy with MMR score
                res = SearchResult(
                    id=sr.id,
                    score=best_mmr_score,
                    distance=sr.distance,
                    vector=sr.vector,
                    metadata=dict(sr.metadata),
                    document=sr.document,
                )
                selected.append(res)
                selected_vectors.append(vec)
            else:
                break

        return selected
