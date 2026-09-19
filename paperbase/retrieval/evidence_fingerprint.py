"""Evidence 的稳定内容指纹。

指纹只依赖可持久化的来源身份与正文，不依赖运行期 E#/R# 编号、排序分数或
本轮 Query，因此可用于 checkpoint 恢复、跨批次去重与 Evidence Analysis 缓存。
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


def normalize_evidence_content(content: str) -> str:
    """压平空白并统一大小写，避免排版差异造成不同指纹。"""
    return " ".join(content.casefold().split())


def build_evidence_fingerprint(evidence: Any) -> str:
    """为 RetrievedEvidence 或兼容对象生成稳定 SHA-256 指纹。"""
    metadata = getattr(evidence, "retrieval_metadata", {}) or {}
    payload = {
        "source_type": getattr(evidence.source_type, "value", evidence.source_type),
        "paper_id": evidence.paper_id,
        "source_chunk_ids": sorted(evidence.source_chunk_ids),
        "overview_field": metadata.get("overview_field"),
        "content": normalize_evidence_content(evidence.content),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
