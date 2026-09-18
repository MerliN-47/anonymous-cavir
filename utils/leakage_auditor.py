"""
[Anonymous] TS-RAG / CAVIR Leakage Auditor & Temporal Isolation Guard
Double-Blind Compliant Modular Architecture
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union


def compute_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Computes exact Dynamic Time Warping (DTW) distance between two 1D sequences.
    Complexity: O(L1 * L2) with space optimization.
    """
    s1 = np.asarray(s1, dtype=np.float64).flatten()
    s2 = np.asarray(s2, dtype=np.float64).flatten()
    n, m = len(s1), len(s2)

    if n == 0 or m == 0:
        return 0.0

    # DP table with 2 rows for memory efficiency
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[0] = np.inf
        for j in range(1, m + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev[:] = curr[:]

    return float(np.sqrt(curr[m]))


def audit_leakage(
    query_embeddings: Union[np.ndarray, torch.Tensor],
    kb_embeddings: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.99
) -> Dict[str, Union[float, int, bool]]:
    """
    Computes maximum pairwise cosine similarity to audit test-KB leakage.
    Flags entries with cosine similarity > threshold (default 0.99).
    """
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if isinstance(kb_embeddings, torch.Tensor):
        kb_embeddings = kb_embeddings.detach().cpu().numpy()

    # Normalize vectors
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=-1, keepdims=True) + 1e-8)
    kb_norm = kb_embeddings / (np.linalg.norm(kb_embeddings, axis=-1, keepdims=True) + 1e-8)

    # Pairwise cosine similarity matrix
    sims = np.dot(q_norm, kb_norm.T)
    max_sim = float(np.max(sims))
    leakage_count = int(np.sum(sims > threshold))

    return {
        "max_cosine_similarity": max_sim,
        "leakage_count": leakage_count,
        "is_contaminated": bool(leakage_count > 0)
    }


def audit_sequence_duplication(
    query_seqs: Union[np.ndarray, torch.Tensor],
    memory_seqs: Union[np.ndarray, torch.Tensor],
    cosine_threshold: float = 0.99,
    dtw_threshold: float = 1e-3
) -> Dict[str, Union[float, int, bool, List[int]]]:
    """
    Dual-criteria leakage audit checking both representation-space cosine similarity
    and raw sequence-space DTW distance.
    """
    if isinstance(query_seqs, torch.Tensor):
        query_seqs = query_seqs.detach().cpu().numpy()
    if isinstance(memory_seqs, torch.Tensor):
        memory_seqs = memory_seqs.detach().cpu().numpy()

    # Flatten trailing dimensions if multi-channel
    q_flat = query_seqs.reshape(query_seqs.shape[0], -1)
    m_flat = memory_seqs.reshape(memory_seqs.shape[0], -1)

    cos_audit = audit_leakage(q_flat, m_flat, threshold=cosine_threshold)

    # Perform spot DTW check on top cosine matches
    q_norm = q_flat / (np.linalg.norm(q_flat, axis=-1, keepdims=True) + 1e-8)
    m_norm = m_flat / (np.linalg.norm(m_flat, axis=-1, keepdims=True) + 1e-8)
    sims = np.dot(q_norm, m_norm.T)

    contaminated_indices = []
    min_dtw_found = float("inf")

    # Sample top matching pairs
    top_pairs = np.argwhere(sims > (cosine_threshold - 0.05))
    for q_idx, m_idx in top_pairs[:50]:  # Cap to first 50 checks for runtime speed
        dist = compute_dtw_distance(q_flat[q_idx], m_flat[m_idx])
        if dist < min_dtw_found:
            min_dtw_found = dist
        if dist < dtw_threshold or sims[q_idx, m_idx] > cosine_threshold:
            contaminated_indices.append(int(m_idx))

    is_contaminated = bool(cos_audit["is_contaminated"] or len(contaminated_indices) > 0)

    return {
        "max_cosine_similarity": cos_audit["max_cosine_similarity"],
        "min_dtw_distance": float(min_dtw_found) if min_dtw_found != float("inf") else 1.0,
        "leakage_count": max(cos_audit["leakage_count"], len(contaminated_indices)),
        "is_contaminated": is_contaminated,
        "contaminated_indices": sorted(list(set(contaminated_indices)))
    }


def apply_leakage_filter(
    query: Union[np.ndarray, torch.Tensor],
    retrieved_neighbors: Union[np.ndarray, torch.Tensor],
    neighbor_vectors: Optional[np.ndarray] = None,
    query_vector: Optional[np.ndarray] = None,
    cos_threshold: float = 0.99,
    dtw_threshold: float = 1e-3
) -> np.ndarray:
    """
    Constructs a boolean acceptance mask (1 = keep, 0 = reject) for retrieved neighbors.
    Rejects any retrieved neighbor that is identical or near-duplicate to query.
    """
    if isinstance(query, torch.Tensor):
        query_np = query.detach().cpu().numpy()
    else:
        query_np = np.asarray(query)

    if isinstance(retrieved_neighbors, torch.Tensor):
        neighbors_np = retrieved_neighbors.detach().cpu().numpy()
    else:
        neighbors_np = np.asarray(retrieved_neighbors)

    k = neighbors_np.shape[0]
    mask = np.ones(k, dtype=np.int32)

    q_flat = query_np.flatten()
    q_norm = q_flat / (np.linalg.norm(q_flat) + 1e-8)

    for i in range(k):
        n_flat = neighbors_np[i].flatten()
        n_norm = n_flat / (np.linalg.norm(n_flat) + 1e-8)
        cos_sim = float(np.dot(q_norm, n_norm))

        if cos_sim >= cos_threshold:
            # Check DTW to confirm true duplicate
            dtw_dist = compute_dtw_distance(q_flat, n_flat)
            if dtw_dist <= dtw_threshold or cos_sim >= 0.9999:
                mask[i] = 0

    return mask


def assert_data_isolation(
    memory_data: Union[np.ndarray, torch.Tensor],
    test_data: Union[np.ndarray, torch.Tensor],
    cosine_threshold: float = 0.99,
    dtw_threshold: float = 1e-3
) -> bool:
    """
    Mathematical guardrail assertion verifying:
    D_memory \cap D_test = \emptyset
    Raises AssertionError if leakage or sequence contamination is detected.
    """
    report = audit_sequence_duplication(
        query_seqs=test_data,
        memory_seqs=memory_data,
        cosine_threshold=cosine_threshold,
        dtw_threshold=dtw_threshold
    )
    if report["is_contaminated"]:
        raise AssertionError(
            f"[Temporal Leakage Guard Violation] D_memory and D_test intersect! "
            f"Max cosine: {report['max_cosine_similarity']:.4f} > {cosine_threshold}, "
            f"Leakage count: {report['leakage_count']}"
        )
    return True
