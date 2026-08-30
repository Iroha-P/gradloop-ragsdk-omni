"""Model-independent multimodal inference contracts."""

from app.multimodal.base import (
    MediaModality,
    MultimodalHealth,
    MultimodalModel,
    OmniAttachment,
    OmniRequest,
    OmniResponse,
    build_attachment,
)
from app.multimodal.minicpmo_client import MiniCPMOClient

__all__ = [
    "MediaModality",
    "MiniCPMOClient",
    "MultimodalHealth",
    "MultimodalModel",
    "OmniAttachment",
    "OmniRequest",
    "OmniResponse",
    "build_attachment",
]
