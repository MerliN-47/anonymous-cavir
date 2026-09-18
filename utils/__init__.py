"""
[Anonymous] TS-RAG / CAVIR Utilities Package
Double-Blind Compliant Modular Architecture
"""

from .run_config import add_all_flags
from .metrics import (
    MSE,
    MAE,
    RMSE,
    MAPE,
    MSPE,
    SMAPE,
    ND,
    RSE,
    CORR,
    pinball_loss,
    CRPS,
    WQL,
    MASE,
    Coverage,
    DieboldMarianoTest,
    metric,
    compute_distributional_metrics
)
from .leakage_auditor import (
    audit_leakage,
    audit_sequence_duplication,
    apply_leakage_filter,
    compute_dtw_distance,
    assert_data_isolation
)
from .faiss_index import (
    TimeSeriesFAISSIndex,
    partition_memory
)

__all__ = [
    "add_all_flags",
    "MSE",
    "MAE",
    "RMSE",
    "MAPE",
    "MSPE",
    "SMAPE",
    "ND",
    "RSE",
    "CORR",
    "pinball_loss",
    "CRPS",
    "WQL",
    "MASE",
    "Coverage",
    "DieboldMarianoTest",
    "metric",
    "compute_distributional_metrics",
    "audit_leakage",
    "audit_sequence_duplication",
    "apply_leakage_filter",
    "compute_dtw_distance",
    "assert_data_isolation",
    "TimeSeriesFAISSIndex",
    "partition_memory"
]
