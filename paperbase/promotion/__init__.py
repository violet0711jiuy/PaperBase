"""Temporary Workspace 提升至正式 Knowledge Base 的增量入库功能。"""

from .service import PromotionResult, PromotionService, promote_workspace

__all__ = ["PromotionResult", "PromotionService", "promote_workspace"]
