#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Covariate & Multimodal Text Adapters.
Implements:
1. CovariateQueryEmbedder: Supervised projection g_phi, h_psi fusing future covariates and static meta-features.
2. ChannelBlockEncoder: Permutation-invariant multivariate cross-channel pooling.
3. MultimodalTextAdapter: Frozen BGE text-event cross-modal adapter for macroeconomic context streams.
Double-Blind Compliant Implementation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("[Anonymous].TS-RAG.covariates_and_text")


class CovariateQueryEmbedder(nn.Module):
    """
    Supervised Covariate Query Embedder (E4):
    Projects time-series embeddings e_ts, future exogenous covariates z_future,
    and static metadata s_meta into a joint query space:
      e_query = LayerNorm(W_x * e_ts(x_q) + W_z * vec(z_{q, t+1:t+H}) + W_s * s_q)
    """
    def __init__(
        self,
        d_model: int = 768,
        cov_dim: int = 16,
        meta_dim: int = 16,
        future_horizon: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.future_horizon = future_horizon

        self.proj_ts = nn.Linear(d_model, d_model)
        self.proj_cov = nn.Linear(future_horizon * cov_dim, d_model)
        self.proj_meta = nn.Linear(meta_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        e_ts: torch.Tensor,                                  # (B, D)
        z_future: Optional[torch.Tensor] = None,             # (B, H, cov_dim)
        s_meta: Optional[torch.Tensor] = None                # (B, meta_dim)
    ) -> torch.Tensor:
        out = self.proj_ts(e_ts)

        if z_future is not None:
            B = z_future.shape[0]
            z_flat = z_future.reshape(B, -1)
            # Match input features
            if z_flat.shape[-1] != self.proj_cov.in_features:
                if z_flat.shape[-1] < self.proj_cov.in_features:
                    pad = torch.zeros(B, self.proj_cov.in_features - z_flat.shape[-1], device=z_future.device)
                    z_flat = torch.cat([z_flat, pad], dim=-1)
                else:
                    z_flat = z_flat[:, :self.proj_cov.in_features]
            out = out + self.proj_cov(z_flat)

        if s_meta is not None:
            B = s_meta.shape[0]
            if s_meta.shape[-1] != self.proj_meta.in_features:
                if s_meta.shape[-1] < self.proj_meta.in_features:
                    pad = torch.zeros(B, self.proj_meta.in_features - s_meta.shape[-1], device=s_meta.device)
                    s_meta = torch.cat([s_meta, pad], dim=-1)
                else:
                    s_meta = s_meta[:, :self.proj_meta.in_features]
            out = out + self.proj_meta(s_meta)

        return self.layer_norm(out)


class ChannelBlockEncoder(nn.Module):
    """
    Permutation-Invariant Multi-Channel Block Encoder (E4):
    Aggregates representations across C multivariate channels.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, channel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        channel_embeddings: (B, C, D)
        Returns block representation: (B, D)
        """
        attn_out, _ = self.mha(channel_embeddings, channel_embeddings, channel_embeddings)
        h = self.norm(channel_embeddings + self.ffn(attn_out))
        return h.mean(dim=1)


class MultimodalTextAdapter(nn.Module):
    """
    Frozen Text Encoder Cross-Modal Adapter (E6):
    Embeds timestamped textual event streams and aligns them with time series embeddings.
    """
    def __init__(self, model_name: str = "bge-large-en", d_model: int = 768):
        super().__init__()
        self.model_name = model_name
        self.d_model = d_model
        self.proj = nn.Linear(1024 if "large" in model_name else 768, d_model)
        self.text_model = None

    def _init_model(self):
        if self.text_model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                hf_name = "BAAI/bge-large-en-v1.5" if "bge" in self.model_name else "sentence-transformers/all-mpnet-base-v2"
                self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
                self.text_model = AutoModel.from_pretrained(hf_name)
                for p in self.text_model.parameters():
                    p.requires_grad = False
                hidden_dim = self.text_model.config.hidden_size
                self.proj = nn.Linear(hidden_dim, self.d_model)
            except Exception as e:
                logger.warning(f"Using lightweight fallback for text encoder ({e})")
                self.text_model = "fallback"

    def forward(self, text_list: List[str], device: str = "cpu") -> torch.Tensor:
        """
        Maps a batch of event strings to aligned embedding vectors (B, D).
        """
        self._init_model()
        if hasattr(self.proj, "to"):
            self.proj = self.proj.to(device)

        if self.text_model != "fallback":
            if hasattr(self.text_model, "to"):
                self.text_model = self.text_model.to(device)
            inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self.text_model(**inputs).last_hidden_state.mean(dim=1)
            return self.proj(out)
        else:
            # Deterministic hash-based representation for offline / fallback evaluation
            embeddings = []
            for text in text_list:
                seed = sum(ord(c) for c in text) % 10000
                g = torch.Generator().manual_seed(seed)
                emb = torch.randn(self.d_model, generator=g, device=device)
                embeddings.append(emb)
            return torch.stack(embeddings, dim=0)


# Alias for backward compatibility
FrozenTextEncoderAdapter = MultimodalTextAdapter
