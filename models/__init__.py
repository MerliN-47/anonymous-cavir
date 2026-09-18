"""
[Anonymous] TS-RAG / CAVIR Models Package
Double-Blind Compliant Modular Architecture
"""

from .backbone_interface import (
    BaseTSFMAdapter,
    ChronosBoltAdapter,
    Chronos2Adapter,
    MoiraiAdapter,
    TimesFMAdapter,
    get_backbone_adapter
)
from .injection_heads import (
    LatentSpaceARM,
    OutputSpaceBarycenter,
    TokenSpaceFormatter,
    UnifiedRetrievalInjector
)
from .controller import SelectiveUtilityGate, DynamicRadiusController
from .conformal_guard import NonExchangeableConformalGuard
from .quotient_canonicalizer import QuotientCanonicalizer
from .covariate_and_text import CovariateQueryEmbedder, MultimodalTextAdapter

__all__ = [
    "BaseTSFMAdapter",
    "ChronosBoltAdapter",
    "Chronos2Adapter",
    "MoiraiAdapter",
    "TimesFMAdapter",
    "get_backbone_adapter",
    "LatentSpaceARM",
    "OutputSpaceBarycenter",
    "TokenSpaceFormatter",
    "UnifiedRetrievalInjector",
    "SelectiveUtilityGate",
    "DynamicRadiusController",
    "NonExchangeableConformalGuard",
    "QuotientCanonicalizer",
    "CovariateQueryEmbedder",
    "MultimodalTextAdapter"
]
