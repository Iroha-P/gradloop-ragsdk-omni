"""Privacy-gated document ingestion package."""

from .models import Sensitivity, SourceSpec
from .pipeline import IngestionPipeline, PipelineConfig
from .source_scope import ScopeDecision, SourceScopePolicy

__all__ = [
    "IngestionPipeline",
    "PipelineConfig",
    "ScopeDecision",
    "Sensitivity",
    "SourceScopePolicy",
    "SourceSpec",
]
