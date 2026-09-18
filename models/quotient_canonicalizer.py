#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Scale-Translation Quotient Canonicalizer.
Implements:
QuotientCanonicalizer: Scale-translation G-invariant quotient projection M = X/G.
Maps continuous time series onto canonical orbit representatives invariant under
the affine group G = R^+ x R (scale and translation shifts).
Double-Blind Compliant Implementation.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class QuotientCanonicalizer(nn.Module):
    """
    Quotient Projection Canonicalizer:
    Projects x in R^L to canonical representative [x] in quotient manifold M = X/G.
    Under action g(x) = a * x + b (a > 0, b in R):
      canonical_x = (x - mu(x)) / (sigma(x) + eps)
    Preserves scale-translation invariants for cross-domain transfer and over-invariance sweeps.
    """
    def __init__(self, mode: str = "full", eps: float = 1e-5):
        super().__init__()
        self.mode = mode  # 'full' (scale + translation), 'translation_only', 'scale_only', 'none'
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        dim: int = -1
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Projects input x to quotient manifold M.
        Returns:
            canonical_x: Invariant normalized tensor with same shape as x
            stats: (mu, sigma) tuple enabling exact inverse group action
        """
        if self.mode == "none":
            loc = torch.zeros_like(x.mean(dim=dim, keepdim=True))
            scale = torch.ones_like(loc)
            return x, (loc, scale)

        # 1. Translation statistic (mean / level)
        loc = torch.nan_to_num(torch.nanmean(x, dim=dim, keepdim=True), nan=0.0)

        # 2. Scale statistic (standard deviation / volatility)
        if self.mode in ["full", "scale_only"]:
            centered = x - loc if self.mode == "full" else x
            scale = torch.nan_to_num(
                torch.sqrt(torch.nanmean(centered.square(), dim=dim, keepdim=True) + self.eps),
                nan=1.0
            )
            # Detect constant series
            is_constant = torch.all(x == x[..., :1], dim=dim, keepdim=True)
            scale = torch.where(is_constant, torch.ones_like(scale), scale)
        else: # translation_only
            scale = torch.ones_like(loc)

        if self.mode == "scale_only":
            loc = torch.zeros_like(scale)

        canonical_x = (x - loc) / scale
        return canonical_x, (loc, scale)

    def inverse(
        self,
        canonical_x: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Applies inverse group action to restore original scale and translation:
          x = canonical_x * sigma + mu
        """
        loc, scale = stats
        return canonical_x * scale + loc
