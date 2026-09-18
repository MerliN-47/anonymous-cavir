#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Selective Controller & Dynamic Radius Adaptation.
Implements:
1. SelectiveUtilityGate: Learned utility gate MLP phi(x_q, d_min) with abstention thresholding.
2. DynamicRadiusController: Adaptive neighborhood radius rho(x_q) conditioned on local manifold curvature.
Double-Blind Compliant Implementation.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveUtilityGate(nn.Module):
    """
    Selective Utility Gate MLP phi(x_q, d_retrieved):
    Learns whether retrieval augmentation improves loss compared to unaugmented zero-shot baseline.
    Produces routing weight g in [0, 1]. When g < tau_abstain, retrieval is bypassed to prevent negative transfer.
    """
    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 128,
        abstain_threshold: float = 0.5,
        mode: str = "softmax"
    ):
        super().__init__()
        self.d_model = d_model
        self.abstain_threshold = abstain_threshold
        self.mode = mode

        # Utility classifier MLP: takes query embedding e_q and summary statistics of retrieved neighbors
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        query_emb: torch.Tensor,               # (B, D)
        min_distance: torch.Tensor,            # (B, 1) or (B,)
        mean_similarity: torch.Tensor          # (B, 1) or (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate_weights: (B, 1) continuous mixing weight in [0, 1]
            abstain_mask: (B, 1) boolean mask where True indicates abstention (bypass RAG)
        """
        if min_distance.dim() == 1:
            min_distance = min_distance.unsqueeze(-1)
        if mean_similarity.dim() == 1:
            mean_similarity = mean_similarity.unsqueeze(-1)

        features = torch.cat([query_emb, min_distance, mean_similarity], dim=-1)
        logits = self.gate_mlp(features)  # (B, 1)
        gate_prob = torch.sigmoid(logits)

        if self.mode == "abstain":
            abstain_mask = (gate_prob < self.abstain_threshold)
            gate_weights = torch.where(abstain_mask, torch.zeros_like(gate_prob), gate_prob)
        else:
            abstain_mask = torch.zeros_like(gate_prob, dtype=torch.bool)
            gate_weights = gate_prob

        return gate_weights, abstain_mask


class DynamicRadiusController(nn.Module):
    """
    Dynamic Neighborhood Radius Adaptation rho(x_q):
    Computes query-conditioned radius parameter rho(x_q) = Softplus(psi(e_q)) + rho_min
    for filtering retrieval neighborhoods based on local manifold density.
    """
    def __init__(self, d_model: int = 768, hidden_dim: int = 64, min_radius: float = 0.05):
        super().__init__()
        self.min_radius = min_radius
        self.radius_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Returns adaptive radius rho(x_q) of shape (B, 1).
        """
        return self.radius_head(query_emb) + self.min_radius
