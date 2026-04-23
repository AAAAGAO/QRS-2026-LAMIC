from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .bm25 import BM25Retriever
from .config import RetrieverWeights
from .data import ApiSample


@dataclass(slots=True)
class RetrievalRow:
    candidate: ApiSample
    bm25_score: float
    semantic_score: float
    sop_score: float
    fused_score: float


def cosine_to_unit_interval(scores: np.ndarray) -> np.ndarray:
    return (scores + 1.0) / 2.0


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, weights: RetrieverWeights) -> None:
        self.bm25 = bm25
        self.weights = weights

    def _embed_samples(self, encoder, samples: list[ApiSample], sop_strings: list[str], device: torch.device) -> torch.Tensor:
        encoder.eval()
        with torch.no_grad():
            if sop_strings:
                return encoder(sop_strings, device)
            return encoder([sample.api for sample in samples], [sample.fragment for sample in samples], device)

    def retrieve(
        self,
        query: ApiSample,
        pool: list[ApiSample],
        pool_semantic: torch.Tensor,
        pool_structural: torch.Tensor,
        semantic_encoder,
        structural_encoder,
        sop_string: str,
        top_k: int,
        device: torch.device,
    ) -> list[RetrievalRow]:
        semantic_query = self._embed_samples(semantic_encoder, [query], [], device)[0]
        structural_query = self._embed_samples(structural_encoder, [], [sop_string], device)[0]
        bm25_result = self.bm25.score(query)

        sem_scores = torch.matmul(pool_semantic, semantic_query).cpu().numpy()
        struct_scores = torch.matmul(pool_structural, structural_query).cpu().numpy()
        sem_unit = cosine_to_unit_interval(sem_scores)
        struct_unit = cosine_to_unit_interval(struct_scores)
        bm25_scores = np.asarray(bm25_result.normalized_scores, dtype=np.float32)

        fused_scores = (
            self.weights.lexical * bm25_scores
            + self.weights.semantic * sem_unit
            + self.weights.structural * struct_unit
        )

        rows: list[RetrievalRow] = []
        for idx, candidate in enumerate(pool):
            if candidate.sample_id == query.sample_id:
                continue
            rows.append(
                RetrievalRow(
                    candidate=candidate,
                    bm25_score=float(bm25_scores[idx]),
                    semantic_score=float(sem_scores[idx]),
                    sop_score=float(struct_scores[idx]),
                    fused_score=float(fused_scores[idx]),
                )
            )
        rows.sort(key=lambda item: item.fused_score, reverse=True)
        return rows[:top_k]


def recall_at_k(rankings: list[list[RetrievalRow]], query_labels: list[int], k: int) -> float:
    hit = 0
    for rows, label in zip(rankings, query_labels, strict=True):
        if any(row.candidate.label == label for row in rows[:k]):
            hit += 1
    return hit / max(len(rankings), 1)


def mean_reciprocal_rank(rankings: list[list[RetrievalRow]], query_labels: list[int]) -> float:
    rr_total = 0.0
    for rows, label in zip(rankings, query_labels, strict=True):
        rr = 0.0
        for rank, row in enumerate(rows, start=1):
            if row.candidate.label == label:
                rr = 1.0 / rank
                break
        rr_total += rr
    return rr_total / max(len(rankings), 1)


def same_api_hit_rate(rankings: list[list[RetrievalRow]], queries: list[ApiSample], k: int) -> float:
    hit = 0
    for rows, query in zip(rankings, queries, strict=True):
        if any(row.candidate.api == query.api for row in rows[:k]):
            hit += 1
    return hit / max(len(rankings), 1)
