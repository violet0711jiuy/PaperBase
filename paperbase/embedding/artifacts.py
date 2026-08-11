"""Step 4 embedding 向量工件的写入、加载与完整性验证。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .base import EmbeddingInput


_ARTIFACT_SCHEMA_VERSION = 1
_VECTORS_FILENAME = "vectors.npy"
_RECORDS_FILENAME = "records.jsonl"
_MANIFEST_FILENAME = "manifest.json"


class EmbeddingArtifactError(RuntimeError):
    """向量工件不完整、顺序不一致或内容不符合 Step 4 契约时抛出。"""


@dataclass(frozen=True)
class EmbeddingRecord:
    """向量矩阵某一行与 SQLite chunk 的可审计映射，不重复保存 embedding 正文。"""

    row_index: int
    chunk_id: str
    paper_id: str
    chunk_index: int
    embedding_text_sha256: str


@dataclass(frozen=True)
class EmbeddingArtifactPaths:
    """一个完整 Step 4 工件组的固定文件路径。"""

    output_dir: Path
    vectors_path: Path
    records_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LoadedEmbeddingArtifact:
    """供未来 Step 5 读取的已验证向量工件。"""

    vectors: np.ndarray
    records: tuple[EmbeddingRecord, ...]
    manifest: dict[str, Any]


def artifact_paths(output_dir: Path) -> EmbeddingArtifactPaths:
    """根据配置目录生成三个工件的唯一、固定命名。"""
    resolved = output_dir.resolve()
    return EmbeddingArtifactPaths(
        output_dir=resolved,
        vectors_path=resolved / _VECTORS_FILENAME,
        records_path=resolved / _RECORDS_FILENAME,
        manifest_path=resolved / _MANIFEST_FILENAME,
    )


def write_embedding_artifact(
    *,
    output_dir: Path,
    inputs: tuple[EmbeddingInput, ...],
    vectors: np.ndarray,
    backend_id: str,
    model_id: str,
    normalized: bool,
    database_path: Path,
) -> EmbeddingArtifactPaths:
    """原子替换一组已验证的向量、行映射和 manifest。

    Step 4 保持向量在 staging 文件中，不把 1024 维数组塞入 SQLite。每条 records.jsonl
    都记录 ``row_index -> chunk_id``，因此未来建立 FAISS 时可以精确给向量分配 ID；
    同时记录 embedding_text 的哈希，防止有人在生成向量后静默修改 SQLite 文本。
    """
    matrix = _validate_matrix(vectors=vectors, input_count=len(inputs), normalized=normalized)
    records = tuple(
        EmbeddingRecord(
            row_index=index,
            chunk_id=item.chunk_id,
            paper_id=item.paper_id,
            chunk_index=item.chunk_index,
            embedding_text_sha256=_text_sha256(item.embedding_text),
        )
        for index, item in enumerate(inputs)
    )
    input_fingerprint = _input_fingerprint(inputs)
    paths = artifact_paths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend_id": backend_id,
        "model_id": model_id,
        "database_path": str(database_path.resolve()),
        "record_count": len(records),
        "dimension": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "normalized": normalized,
        "input_fingerprint_sha256": input_fingerprint,
        "vectors_file": _VECTORS_FILENAME,
        "records_file": _RECORDS_FILENAME,
    }

    # 三个临时文件在同一目录创建，随后用 os.replace 覆盖正式文件。读者只接受 manifest
    # 与向量/records 一致的一整组文件；任何异常发生时，临时文件会被清理，旧工件可继续使用。
    run_token = uuid4().hex
    temporary_paths = {
        "vectors": paths.output_dir / f".{_VECTORS_FILENAME}.{run_token}.tmp",
        "records": paths.output_dir / f".{_RECORDS_FILENAME}.{run_token}.tmp",
        "manifest": paths.output_dir / f".{_MANIFEST_FILENAME}.{run_token}.tmp",
    }
    try:
        with temporary_paths["vectors"].open("wb") as output:
            np.save(output, matrix, allow_pickle=False)
        temporary_paths["records"].write_text(
            "".join(
                json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        # manifest 中保存两个数据文件的哈希。若进程恰好在替换数据文件后异常，旧 manifest
        # 会与新数据文件哈希不一致，读取器会明确拒绝，而不会把混合批次交给 FAISS。
        manifest["vectors_sha256"] = _file_sha256(temporary_paths["vectors"])
        manifest["records_sha256"] = _file_sha256(temporary_paths["records"])
        temporary_paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # 先替换数据文件，最后替换 manifest。未来读取器以 manifest 为入口，因而它出现时
        # 对应的 records 和 vectors 已经就位；这比逐个直接覆写正式文件更不易留下半成品。
        os.replace(temporary_paths["vectors"], paths.vectors_path)
        os.replace(temporary_paths["records"], paths.records_path)
        os.replace(temporary_paths["manifest"], paths.manifest_path)
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
    return paths


def load_embedding_artifact(output_dir: Path) -> LoadedEmbeddingArtifact:
    """读取并交叉校验 Step 4 工件，供 Step 5 建索引前使用。"""
    paths = artifact_paths(output_dir)
    if not paths.manifest_path.is_file():
        raise EmbeddingArtifactError(f"Embedding manifest not found: {paths.manifest_path}")
    try:
        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EmbeddingArtifactError("Embedding manifest is not valid JSON.") from error
    if not isinstance(manifest, dict):
        raise EmbeddingArtifactError("Embedding manifest must be a JSON object.")
    if manifest.get("artifact_schema_version") != _ARTIFACT_SCHEMA_VERSION:
        raise EmbeddingArtifactError(
            "Unsupported embedding artifact schema version: "
            f"{manifest.get('artifact_schema_version')!r}"
        )
    if manifest.get("vectors_file") != _VECTORS_FILENAME or manifest.get("records_file") != _RECORDS_FILENAME:
        raise EmbeddingArtifactError("Embedding manifest uses unexpected artifact filenames.")
    if not paths.vectors_path.is_file() or not paths.records_path.is_file():
        raise EmbeddingArtifactError("Embedding artifact is missing vectors.npy or records.jsonl.")
    if (
        manifest.get("vectors_sha256") != _file_sha256(paths.vectors_path)
        or manifest.get("records_sha256") != _file_sha256(paths.records_path)
    ):
        raise EmbeddingArtifactError(
            "Embedding artifact files do not match the manifest; rerun Step 4 to rebuild them."
        )

    try:
        vectors = np.load(paths.vectors_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise EmbeddingArtifactError("Embedding vectors.npy cannot be loaded safely.") from error
    raw_records = [line for line in paths.records_path.read_text(encoding="utf-8").splitlines() if line]
    try:
        records = tuple(EmbeddingRecord(**json.loads(line)) for line in raw_records)
    except (TypeError, json.JSONDecodeError) as error:
        raise EmbeddingArtifactError("Embedding records.jsonl contains an invalid record.") from error

    matrix = _validate_matrix(
        vectors=vectors,
        input_count=len(records),
        normalized=bool(manifest.get("normalized")),
    )
    if manifest.get("record_count") != len(records) or manifest.get("dimension") != matrix.shape[1]:
        raise EmbeddingArtifactError("Embedding manifest count or dimension disagrees with artifact files.")
    if [record.row_index for record in records] != list(range(len(records))):
        raise EmbeddingArtifactError("Embedding record row_index values must be contiguous from 0.")
    if len({record.chunk_id for record in records}) != len(records):
        raise EmbeddingArtifactError("Embedding records contain duplicate chunk_id values.")
    return LoadedEmbeddingArtifact(vectors=matrix, records=records, manifest=manifest)


def _validate_matrix(*, vectors: np.ndarray, input_count: int, normalized: bool) -> np.ndarray:
    """验证矩阵可安全交给未来的 FAISS IndexFlatIP，而不是仅依赖 numpy 能加载。"""
    matrix = np.asarray(vectors)
    if matrix.dtype != np.float32:
        raise EmbeddingArtifactError(f"Embedding vectors must use float32, got {matrix.dtype}.")
    if matrix.ndim != 2 or matrix.shape[0] != input_count or matrix.shape[1] < 1:
        raise EmbeddingArtifactError(
            "Embedding matrix shape does not match records: "
            f"{matrix.shape!r}, records={input_count}."
        )
    if not np.all(np.isfinite(matrix)):
        raise EmbeddingArtifactError("Embedding matrix contains non-finite values.")
    if normalized:
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise EmbeddingArtifactError("Embedding matrix is declared normalized but has non-unit rows.")
    return np.ascontiguousarray(matrix)


def _input_fingerprint(inputs: tuple[EmbeddingInput, ...]) -> str:
    """为“顺序 + chunk 身份 + 实际 embedding 输入”生成稳定快照哈希。"""
    digest = hashlib.sha256()
    for index, item in enumerate(inputs):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_text_sha256(item.embedding_text).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    """只记录 embedding 输入的完整性哈希，不在 records 文件再次复制长文本。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """以流式方式计算工件文件校验和，避免大向量矩阵额外复制进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
