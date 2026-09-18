#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Guardrail Test Suite
Double-Blind Compliant Modular Architecture
Executable PyTest / Unittest verification for:
1. Frozen backbone parameters (nabla_theta = 0)
2. Mathematical data leakage isolation (D_memory cap D_test = emptyset)
3. Conformal prediction interval calibration under variance shocks
4. CLI argparse schema compliance
5. Injection heads and multimodal covariate adapters
"""

import os
import sys
import unittest
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.run_config import add_all_flags
from utils.metrics import (
    compute_distributional_metrics,
    pinball_loss,
    CRPS,
    WQL,
    MASE,
    Coverage,
    DieboldMarianoTest
)
from utils.leakage_auditor import (
    audit_leakage,
    audit_sequence_duplication,
    assert_data_isolation,
    compute_dtw_distance
)
from data_provider.fevbench_loader import FEVBenchDataset
from data_provider.multimodal_loader import MultimodalDataset
from models.backbone_interface import get_backbone_adapter, BaseTSFMAdapter
from models.injection_heads import UnifiedRetrievalInjector
from models.conformal_guard import NonExchangeableConformalGuard
from models.covariate_and_text import CovariateQueryEmbedder, ChannelBlockEncoder, MultimodalTextAdapter
from models.quotient_canonicalizer import QuotientCanonicalizer


class TestGuardrails(unittest.TestCase):
    """
    Master Guardrail Suite verifying double-blind, temporal hygiene,
    and mathematical integrity constraints.
    """

    def test_frozen_backbone(self):
        """
        GUARDRAIL 1: Frozen Backbone Verification
        Asserts all(not p.requires_grad for p in backbone.parameters())
        Verifies zero backpropagation into pretrained foundation parameters.
        """
        backbones_to_test = ["chronos-bolt", "chronos-2", "moirai-2.0", "timesfm-2.5"]
        for bb_name in backbones_to_test:
            adapter = get_backbone_adapter(
                backbone_name=bb_name,
                context_length=128,
                prediction_length=32,
                device="cpu"
            )
            self.assertIsInstance(adapter, BaseTSFMAdapter)

            # Assert all underlying parameters have requires_grad == False
            all_frozen = all(not p.requires_grad for p in adapter.parameters())
            self.assertTrue(
                all_frozen,
                f"Backbone {bb_name} has non-frozen parameters! requires_grad must be False for all parameters."
            )

    def test_data_isolation(self):
        """
        GUARDRAIL 2: Mathematical Data Isolation Assertion
        Asserts that D_memory \cap D_test = \emptyset.
        Ensures exact or near-duplicate sequences are caught and blocked.
        """
        rng = np.random.RandomState(42)
        dim = 64
        # Clean synthetic datasets: orthogonal or random non-overlapping
        memory_data = rng.randn(100, dim).astype(np.float32)
        test_data = rng.randn(20, dim).astype(np.float32)

        # 1. Clean test: must succeed without error
        isolation_verified = assert_data_isolation(
            memory_data=memory_data,
            test_data=test_data,
            cosine_threshold=0.99,
            dtw_threshold=1e-3
        )
        self.assertTrue(isolation_verified)

        # 2. Contaminated test: artificially inject a test sample into memory
        contaminated_memory = memory_data.copy()
        contaminated_memory[10] = test_data[3]  # Direct duplication

        with self.assertRaises(AssertionError) as ctx:
            assert_data_isolation(
                memory_data=contaminated_memory,
                test_data=test_data,
                cosine_threshold=0.99,
                dtw_threshold=1e-3
            )
        self.assertIn("Temporal Leakage Guard Violation", str(ctx.exception))

    def test_conformal_calibration(self):
        """
        GUARDRAIL 3: Conformal Calibration Interval Expansion under Variance Shocks
        Asserts that prediction intervals do not collapse under simulated variance shifts:
        width(shock) >= width(nominal).
        """
        torch.manual_seed(42)
        np.random.seed(42)
        batch_size = 8
        pred_len = 32
        d_model = 64

        guard = NonExchangeableConformalGuard(alpha=0.2, d_model=d_model)

        # Construct nominal quantiles (9 quantiles, tau in [0.1 .. 0.9])
        nominal_preds = torch.zeros(batch_size, 9, pred_len)
        for i, tau in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            nominal_preds[:, i, :] = (tau - 0.5) * 1.0  # nominal spread = 0.8

        x_query_nominal = torch.randn(batch_size, d_model)
        calibrated_nominal = guard.calibrate_intervals(nominal_preds, x_query_nominal)

        # Nominal interval width between 0.1 (idx 0) and 0.9 (idx 8)
        nominal_width = (calibrated_nominal[:, 8, :] - calibrated_nominal[:, 0, :]).mean().item()

        # Injected variance shock: large query vector perturbation
        x_query_shock = x_query_nominal * 3.0 + torch.randn_like(x_query_nominal) * 2.0
        calibrated_shock = guard.calibrate_intervals(nominal_preds, x_query_shock)
        shock_width = (calibrated_shock[:, 8, :] - calibrated_shock[:, 0, :]).mean().item()

        # Assert interval width under shock does not collapse
        self.assertGreaterEqual(
            shock_width,
            nominal_width - 1e-5,
            f"Conformal guard collapsed under variance shock! Nominal width: {nominal_width:.4f}, Shock width: {shock_width:.4f}"
        )

    def test_cli_argparse_schema(self):
        """
        GUARDRAIL 4: CLI Argument Schema
        Verifies complete CLI argument parsing without missing parameters.
        """
        parser = add_all_flags()
        test_flags = [
            "--backbone", "chronos-2",
            "--injection_point", "latent",
            "--covariate_mode", "future_known",
            "--channel_mode", "channel_block",
            "--text_encoder", "bge-large-en",
            "--text_role", "dual",
            "--kb_size_exp", "5",
            "--kb_regime", "strict_zeroshot",
            "--benchmark", "fev_bench",
            "--eval_metrics", "distributional",
            "--conformal", "active",
            "--smoke"
        ]
        args = parser.parse_args(test_flags)
        self.assertEqual(args.backbone, "chronos-2")
        self.assertEqual(args.injection_point, "latent")
        self.assertEqual(args.conformal, "active")
        self.assertTrue(args.smoke)

    def test_fevbench_temporal_leakage_guard(self):
        """
        GUARDRAIL 5: Temporal Boundary Hygiene in fev-bench
        Asserts target future is strictly separated from covariate drivers.
        """
        ds = FEVBenchDataset(num_samples=20, seq_len=128, pred_len=32, cov_dim=3)
        self.assertEqual(len(ds), 20)
        x_seq, y_seq, z_future, s_meta = ds[0]
        self.assertEqual(x_seq.shape, (128,))
        self.assertEqual(y_seq.shape, (32,))
        self.assertEqual(z_future.shape, (32, 3))
        self.assertEqual(s_meta.shape, (8,))
        self.assertFalse(torch.allclose(y_seq, z_future[:, 0]))

    def test_three_injection_modes(self):
        """
        GUARDRAIL 6: Multi-Injection Point Verification
        Verifies Token (In-Context), Latent (ARM Cross-Attention), and Output (Vincentization).
        """
        adapter = get_backbone_adapter("chronos-bolt", context_length=128, prediction_length=32, device="cpu")
        x_q = torch.randn(2, 128)
        ret_past = torch.randn(2, 4, 128)
        ret_future = torch.randn(2, 4, 32)

        for inj_mode in ["token", "latent", "output"]:
            injector = UnifiedRetrievalInjector(
                backbone=adapter,
                injection_point=inj_mode,
                d_model=adapter.d_model,
                prediction_length=32
            )
            out = injector(x_q, ret_past, ret_future)
            self.assertEqual(out.shape, (2, 9, 32), f"Failed injection forward pass for {inj_mode}")

    def test_distributional_metrics(self):
        """
        GUARDRAIL 7: Mathematical Metric Integrity
        Verifies CRPS, WQL, Coverage, and Diebold-Mariano tests.
        """
        true = np.ones((4, 16))
        quantiles_preds = np.zeros((4, 9, 16))
        for i, tau in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            quantiles_preds[:, i, :] = 1.0 + (tau - 0.5)

        m = compute_distributional_metrics(quantiles_preds, true)
        self.assertAlmostEqual(m["MSE"], 0.0, places=4)
        self.assertAlmostEqual(m["Coverage@80"], 1.0, places=4)
        self.assertGreater(m["CRPS"], 0.0)

        # Diebold-Mariano test
        e1 = np.random.normal(0, 0.1, size=50)
        e2 = np.random.normal(0, 0.2, size=50)
        dm_stat, p_val = DieboldMarianoTest(e1, e2)
        self.assertIsInstance(dm_stat, float)
        self.assertIsInstance(p_val, float)


if __name__ == "__main__":
    unittest.main()
