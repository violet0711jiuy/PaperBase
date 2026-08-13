"""Step 5：全局 FAISS IndexIDMap2(IndexFlatIP) 的首次建库与一致性校验。"""

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

from paperbase.config import IndexingSettings
from paperbase.database import MetadataDatabase, MetadataDatabaseError
from paperbase.embedding.artifacts import LoadedEmbeddingArtifact, load_embedding_artifact


_INDEX_SCHEMA_VERSION = 1
_JOURNAL_SCHEMA_VERSION = 1


class FaissIndexError(RuntimeError):
    """FAISS 索引、SQLite 映射或跨文件发布状态不一致时抛出。"""


@dataclass(frozen=True)
class VectorAssignment:
    """一个 staging 向量应写入哪个正式 vector_id 的不可变映射。"""

    chunk_id: str
    vector_id: int
    embedding_text_sha256: str


@dataclass(frozen=True)
class IndexBuildSummary:
    """首次建库或恢复后的可展示摘要。"""

    vector_count: int
    dimension: int
    vector_id_min: int
    vector_id_max: int
    index_path: Path
    manifest_path: Path
    recovered_pending_publish: bool


class FaissIndexStore:
    """管理唯一正式 FAISS 索引文件及其与 SQLite 的映射契约。

    当前类只实现“空知识库的首次全量建库”。这与已完成的 Step 4 快照正好匹配，并避免在
    增量上传服务尚未设计、验证前就混入新的生命周期分支。后续增量加入将复用同一份
    ``vector_id -> chunk_id`` 契约与 manifest 校验，而不是另建每篇论文一个索引。
    """

    backend_id = "faiss_flat_ip"

    def __init__(self, settings: IndexingSettings) -> None:
        self._settings = settings
        self._index_path = settings.index_path.resolve()
        self._manifest_path = settings.manifest_path.resolve()
        self._journal_path = self._index_path.with_suffix(
            self._index_path.suffix + ".pending.json"
        )
        if self._index_path == self._manifest_path:
            raise ValueError("index_path and manifest_path must be different files.")

    @property
    def index_path(self) -> Path:
        """返回正式 FAISS 二进制索引路径。"""
        return self._index_path

    @property
    def manifest_path(self) -> Path:
        """返回与正式索引配套的 JSON 清单路径。"""
        return self._manifest_path

    def build_initial_index(
        self,
        *,
        database: MetadataDatabase,
        embedding_output_dir: Path,
        replace_existing: bool = False,
    ) -> IndexBuildSummary:
        """将已验证的 Step 4 工件写为第一次正式全库 FAISS 索引。

        发布顺序受 pending journal 保护：先写候选索引和 journal，再在一个 SQLite 事务中
        分配 ID，最后原子发布 FAISS 文件与 manifest。FAISS 和 SQLite 不能组成同一个事务，
        所以进程中断后下次运行会依据 journal 恢复发布或安全清理候选文件，而不是猜测状态。
        """
        recovered = self.recover_pending_publish(database)
        if recovered:
            # 恢复已完成时，SQLite 已拥有 vector_id，不能再走“首次分配”分支；直接以相同
            # staging 工件交叉验证并把恢复标记返回给调用方。
            verified = self.verify_against_sqlite(
                database=database,
                embedding_output_dir=embedding_output_dir,
            )
            return IndexBuildSummary(
                vector_count=verified.vector_count,
                dimension=verified.dimension,
                vector_id_min=verified.vector_id_min,
                vector_id_max=verified.vector_id_max,
                index_path=verified.index_path,
                manifest_path=verified.manifest_path,
                recovered_pending_publish=True,
            )
        state = database.vector_index_state()
        if state.vectorized_chunk_count:
            raise FaissIndexError(
                "Initial FAISS build requires SQLite chunks.vector_id to be empty. "
                "Use the future incremental index workflow for an existing index."
            )
        if state.total_chunk_count == 0:
            raise FaissIndexError("Cannot build a FAISS index without SQLite chunks.")
        if (self._index_path.exists() or self._manifest_path.exists()) and not replace_existing:
            raise FaissIndexError(
                "A FAISS index or manifest already exists while SQLite has no vector IDs. "
                "Refusing to overwrite an unverified file."
            )

        artifact = load_embedding_artifact(embedding_output_dir)
        _validate_artifact_for_flat_ip(artifact)
        records = tuple(
            (record.chunk_id, record.embedding_text_sha256)
            for record in artifact.records
        )
        try:
            database.validate_embedding_artifact_records(records)
        except MetadataDatabaseError as error:
            raise FaissIndexError(str(error)) from error
        assignments = tuple(
            VectorAssignment(
                chunk_id=record.chunk_id,
                # 初始索引从 1 开始分配；0 在很多外部系统中常被当成未设置值，避开它能让
                # 后续日志、调试和 UI 展示更直观。它与 chunk_index 和 vectors.npy 行号无关。
                vector_id=index + 1,
                embedding_text_sha256=record.embedding_text_sha256,
            )
            for index, record in enumerate(artifact.records)
        )
        operation_id = uuid4().hex
        candidate_index_path = self._candidate_index_path(operation_id)
        journal_written = False
        try:
            _write_faiss_index(
                path=candidate_index_path,
                vectors=artifact.vectors,
                vector_ids=np.asarray(
                    [assignment.vector_id for assignment in assignments], dtype=np.int64
                ),
            )
            candidate_index_sha256 = _file_sha256(candidate_index_path)
            manifest = _build_manifest(
                artifact=artifact,
                assignments=assignments,
                index_sha256=candidate_index_sha256,
            )
            journal = {
                "journal_schema_version": _JOURNAL_SCHEMA_VERSION,
                "operation_id": operation_id,
                "phase": "prepared",
                "index_path": str(self._index_path),
                "manifest_path": str(self._manifest_path),
                "candidate_index_path": str(candidate_index_path),
                "assignments": [asdict(assignment) for assignment in assignments],
                "manifest": manifest,
            }
            _write_json_atomic(self._journal_path, journal)
            journal_written = True

            try:
                database.assign_vector_ids(
                    [
                        (
                            assignment.chunk_id,
                            assignment.vector_id,
                            assignment.embedding_text_sha256,
                        )
                        for assignment in assignments
                    ]
                )
            except MetadataDatabaseError as error:
                raise FaissIndexError(str(error)) from error
            journal["phase"] = "database_committed"
            _write_json_atomic(self._journal_path, journal)

            self._publish_from_journal(database=database, journal=journal)
            self._journal_path.unlink(missing_ok=True)
        except Exception:
            # journal 已落盘时不能删除候选索引：其中包含恢复 SQLite 已提交映射所需的唯一证据。
            # journal 尚未写入时，候选文件对任何人都不可见，可以立即清理。
            if not journal_written:
                candidate_index_path.unlink(missing_ok=True)
            raise

        return IndexBuildSummary(
            vector_count=len(assignments),
            dimension=int(artifact.vectors.shape[1]),
            vector_id_min=assignments[0].vector_id,
            vector_id_max=assignments[-1].vector_id,
            index_path=self._index_path,
            manifest_path=self._manifest_path,
            recovered_pending_publish=recovered,
        )

    def recover_pending_publish(self, database: MetadataDatabase) -> bool:
        """恢复被中断的首次建库发布；无 journal 时返回 ``False``。"""
        if not self._journal_path.is_file():
            return False
        journal = _load_journal(self._journal_path)
        if Path(journal["index_path"]).resolve() != self._index_path:
            raise FaissIndexError("Pending index journal targets a different index_path.")
        if Path(journal["manifest_path"]).resolve() != self._manifest_path:
            raise FaissIndexError("Pending index journal targets a different manifest_path.")
        assignments = tuple(
            VectorAssignment(**item) for item in journal["assignments"]
        )
        _validate_assignments(assignments)
        actual_mapping = {
            int(row["vector_id"]): str(row["chunk_id"])
            for row in database.vector_id_mapping()
        }
        expected_mapping = {
            assignment.vector_id: assignment.chunk_id for assignment in assignments
        }
        if not actual_mapping:
            # SQLite 未进入提交阶段：没有正式映射可恢复，安全丢弃候选文件并让用户重跑。
            Path(journal["candidate_index_path"]).unlink(missing_ok=True)
            self._journal_path.unlink(missing_ok=True)
            return False
        if actual_mapping != expected_mapping:
            raise FaissIndexError(
                "Pending index journal does not match SQLite vector_id mappings; "
                "manual recovery is required."
            )
        self._publish_from_journal(database=database, journal=journal)
        self._journal_path.unlink(missing_ok=True)
        return True

    def verify_against_sqlite(
        self,
        *,
        database: MetadataDatabase,
        embedding_output_dir: Path,
    ) -> IndexBuildSummary:
        """交叉验证正式 FAISS、SQLite 映射与 Step 4 输入快照，没有写入动作。"""
        if not self._index_path.is_file() or not self._manifest_path.is_file():
            raise FaissIndexError("FAISS index or manifest is missing.")
        artifact = load_embedding_artifact(embedding_output_dir)
        _validate_artifact_for_flat_ip(artifact)
        records = tuple(
            (record.chunk_id, record.embedding_text_sha256)
            for record in artifact.records
        )
        try:
            database.validate_embedding_artifact_records(records)
        except MetadataDatabaseError as error:
            raise FaissIndexError(str(error)) from error
        mappings = database.vector_id_mapping()
        if len(mappings) != len(artifact.records):
            raise FaissIndexError(
                "FAISS verification requires every current SQLite chunk to have one vector_id."
            )
        text_sha_by_chunk_id = dict(records)
        try:
            assignments = tuple(
                VectorAssignment(
                    chunk_id=str(row["chunk_id"]),
                    vector_id=int(row["vector_id"]),
                    embedding_text_sha256=text_sha_by_chunk_id[str(row["chunk_id"])],
                )
                for row in mappings
            )
        except KeyError as error:
            raise FaissIndexError(
                "SQLite vector mapping refers to a chunk not present in the embedding artifact."
            ) from error
        _validate_assignments(assignments)
        manifest = _load_manifest(self._manifest_path)
        _validate_manifest(
            manifest=manifest,
            assignments=assignments,
            dimension=int(artifact.vectors.shape[1]),
            index_path=self._index_path,
        )
        _validate_index_file(
            index_path=self._index_path,
            expected_dimension=int(artifact.vectors.shape[1]),
            expected_vector_ids={assignment.vector_id for assignment in assignments},
        )
        return IndexBuildSummary(
            vector_count=len(assignments),
            dimension=int(artifact.vectors.shape[1]),
            vector_id_min=min(assignment.vector_id for assignment in assignments),
            vector_id_max=max(assignment.vector_id for assignment in assignments),
            index_path=self._index_path,
            manifest_path=self._manifest_path,
            recovered_pending_publish=False,
        )

    def verify_published_index(self, *, database: MetadataDatabase) -> IndexBuildSummary:
        """只依赖正式 FAISS 与 SQLite 验证线上检索映射，不读取 Step 4 staging 工件。

        正式问答不能依赖 ``vectors.npy`` 或 ``records.jsonl``。SQLite 仍保存当前完整的
        ``embedding_text``，因此可重新计算 assignment 指纹，证明 manifest、FAISS ID 集合与
        当前业务数据来自同一版本。
        """
        if not self._index_path.is_file() or not self._manifest_path.is_file():
            raise FaissIndexError("FAISS index or manifest is missing.")
        state = database.vector_index_state()
        if state.total_chunk_count == 0 or state.vectorized_chunk_count != state.total_chunk_count:
            raise FaissIndexError(
                "Published FAISS verification requires every current SQLite chunk to have vector_id."
            )
        assignments = tuple(
            VectorAssignment(
                chunk_id=str(row["chunk_id"]),
                vector_id=int(row["vector_id"]),
                embedding_text_sha256=hashlib.sha256(
                    str(row["embedding_text"]).encode("utf-8")
                ).hexdigest(),
            )
            for row in database.vector_id_assignments()
        )
        _validate_assignments(assignments)
        manifest = _load_manifest(self._manifest_path)
        dimension = int(manifest.get("dimension", 0))
        if dimension < 1:
            raise FaissIndexError("FAISS manifest has an invalid dimension.")
        _validate_manifest(
            manifest=manifest,
            assignments=assignments,
            dimension=dimension,
            index_path=self._index_path,
        )
        _validate_index_file(
            index_path=self._index_path,
            expected_dimension=dimension,
            expected_vector_ids={assignment.vector_id for assignment in assignments},
        )
        return IndexBuildSummary(
            vector_count=len(assignments),
            dimension=dimension,
            vector_id_min=min(assignment.vector_id for assignment in assignments),
            vector_id_max=max(assignment.vector_id for assignment in assignments),
            index_path=self._index_path,
            manifest_path=self._manifest_path,
            recovered_pending_publish=False,
        )

    def load_for_search(self, *, database: MetadataDatabase) -> Any:
        """验证正式映射后加载 FAISS 索引；调用方无需接触 staging 工件。"""
        self.verify_published_index(database=database)
        return _read_faiss_index(self._index_path)

    def _publish_from_journal(
        self,
        *,
        database: MetadataDatabase,
        journal: dict[str, Any],
    ) -> None:
        """确认 SQLite 已完整提交后，发布候选索引和与之匹配的 manifest。"""
        assignments = tuple(
            VectorAssignment(**item) for item in journal["assignments"]
        )
        expected_vector_ids = {assignment.vector_id for assignment in assignments}
        actual_mapping = {
            int(row["vector_id"]): str(row["chunk_id"])
            for row in database.vector_id_mapping()
        }
        expected_mapping = {
            assignment.vector_id: assignment.chunk_id for assignment in assignments
        }
        if actual_mapping != expected_mapping:
            raise FaissIndexError(
                "Cannot publish FAISS because SQLite does not contain the expected complete mapping."
            )

        manifest = journal["manifest"]
        dimension = int(manifest["dimension"])
        candidate_index_path = Path(journal["candidate_index_path"])
        # 若上一次中断已经移动了候选文件，正式索引也可作为恢复来源；两者都必须通过同一套
        # ID 集合和维度验证，绝不只因为文件存在就信任它。
        if not _index_file_matches(
            self._index_path,
            expected_dimension=dimension,
            expected_vector_ids=expected_vector_ids,
        ):
            if not _index_file_matches(
                candidate_index_path,
                expected_dimension=dimension,
                expected_vector_ids=expected_vector_ids,
            ):
                raise FaissIndexError(
                    "Neither the published nor candidate FAISS file matches the pending journal."
                )
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate_index_path, self._index_path)

        # 只有正式索引文件的校验和与 journal 中的候选索引一致，才允许公开新的 manifest。
        if _file_sha256(self._index_path) != manifest.get("index_sha256"):
            raise FaissIndexError("Published FAISS file checksum does not match the pending journal.")
        _write_json_atomic(self._manifest_path, manifest)
        _validate_manifest(
            manifest=manifest,
            assignments=assignments,
            dimension=dimension,
            index_path=self._index_path,
        )

    def _candidate_index_path(self, operation_id: str) -> Path:
        """候选索引始终写在正式文件同目录，确保 os.replace 为同一文件系统内原子替换。"""
        return self._index_path.with_name(
            f".{self._index_path.name}.{operation_id}.candidate"
        )


def _write_faiss_index(*, path: Path, vectors: np.ndarray, vector_ids: np.ndarray) -> None:
    """构建并自校验 ``IndexIDMap2(IndexFlatIP)`` 候选文件。"""
    if vectors.dtype != np.float32 or vectors.ndim != 2:
        raise FaissIndexError("FAISS input vectors must be a two-dimensional float32 matrix.")
    if vector_ids.dtype != np.int64 or vector_ids.shape != (vectors.shape[0],):
        raise FaissIndexError("FAISS vector IDs must be an int64 array aligned with vectors.")
    faiss = _import_faiss()
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(vectors.shape[1])))
    index.add_with_ids(np.ascontiguousarray(vectors), np.ascontiguousarray(vector_ids))
    if int(index.ntotal) != vectors.shape[0]:
        raise FaissIndexError("FAISS accepted an unexpected number of vectors.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # FAISS 1.15 的 Windows 文件路径 API 无法稳定处理含中文用户名的临时目录。使用
    # serialize_index / Python Path 写文件可规避该编码边界；索引本身仍是标准 FAISS 二进制。
    # MVP 知识库规模较小，序列化时的短暂内存副本可以接受；大规模索引时再评估回调式 I/O。
    serialized_index = faiss.serialize_index(index)
    path.write_bytes(np.asarray(serialized_index, dtype=np.uint8).tobytes())
    _validate_index_file(
        index_path=path,
        expected_dimension=int(vectors.shape[1]),
        expected_vector_ids={int(vector_id) for vector_id in vector_ids},
    )


def _build_manifest(
    *,
    artifact: LoadedEmbeddingArtifact,
    assignments: tuple[VectorAssignment, ...],
    index_sha256: str,
) -> dict[str, Any]:
    """生成正式索引清单；不复制向量和全文，只记录可交叉验证的指纹。"""
    _validate_assignments(assignments)
    return {
        "index_schema_version": _INDEX_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend_id": "faiss_flat_ip",
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "metric": "inner_product",
        "dimension": int(artifact.vectors.shape[1]),
        "vector_count": len(assignments),
        "vector_id_min": assignments[0].vector_id,
        "vector_id_max": assignments[-1].vector_id,
        "assignment_fingerprint_sha256": _assignment_fingerprint(assignments),
        "index_sha256": index_sha256,
        "embedding_artifact": {
            "model_id": artifact.manifest["model_id"],
            "input_fingerprint_sha256": artifact.manifest["input_fingerprint_sha256"],
            "normalized": artifact.manifest["normalized"],
            "record_count": artifact.manifest["record_count"],
        },
    }


def _validate_artifact_for_flat_ip(artifact: LoadedEmbeddingArtifact) -> None:
    """IndexFlatIP 只有在文档向量已单位化时才表达本项目要求的余弦相似度。"""
    if not artifact.manifest.get("normalized"):
        raise FaissIndexError(
            "FAISS IndexFlatIP requires a normalized embedding artifact for cosine retrieval."
        )
    if artifact.vectors.dtype != np.float32 or artifact.vectors.ndim != 2:
        raise FaissIndexError("Embedding artifact must contain a two-dimensional float32 matrix.")
    if len(artifact.records) != artifact.vectors.shape[0]:
        raise FaissIndexError("Embedding artifact record count does not match vector count.")


def _validate_assignments(assignments: tuple[VectorAssignment, ...]) -> None:
    """阻止重复 ID、重复 chunk 或无效 ID 在触及 FAISS/SQLite 前进入发布流程。"""
    if not assignments:
        raise FaissIndexError("Vector assignment set must not be empty.")
    chunk_ids = [assignment.chunk_id for assignment in assignments]
    vector_ids = [assignment.vector_id for assignment in assignments]
    if len(set(chunk_ids)) != len(chunk_ids) or len(set(vector_ids)) != len(vector_ids):
        raise FaissIndexError("Vector assignments must have unique chunk_id and vector_id values.")
    if min(vector_ids) < 1:
        raise FaissIndexError("FAISS vector IDs must be positive integers.")


def _validate_index_file(
    *,
    index_path: Path,
    expected_dimension: int,
    expected_vector_ids: set[int],
) -> None:
    """读取刚写入或准备发布的索引，验证维度、数量与真实 ID 集合。"""
    if not _index_file_matches(
        index_path,
        expected_dimension=expected_dimension,
        expected_vector_ids=expected_vector_ids,
    ):
        raise FaissIndexError(
            "FAISS index does not match its expected dimension, count, or vector ID set."
        )


def _index_file_matches(
    index_path: Path,
    *,
    expected_dimension: int,
    expected_vector_ids: set[int],
) -> bool:
    """以布尔值检测某个候选/正式文件是否正好是预期的 IndexIDMap2。"""
    if not index_path.is_file():
        return False
    try:
        faiss = _import_faiss()
        # 同写入路径一样，避免调用 FAISS 的 Windows 路径 API，以支持含中文的用户目录。
        index = _read_faiss_index(index_path)
        stored_ids = faiss.vector_to_array(index.id_map)
    except Exception:
        return False
    return (
        int(index.d) == expected_dimension
        and int(index.ntotal) == len(expected_vector_ids)
        and len(stored_ids) == len(expected_vector_ids)
        and {int(vector_id) for vector_id in stored_ids} == expected_vector_ids
    )


def _read_faiss_index(index_path: Path) -> Any:
    """通过 Python 文件 I/O 读取标准 FAISS 二进制，避开 Windows 非 ASCII 路径限制。"""
    faiss = _import_faiss()
    serialized_index = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
    return faiss.deserialize_index(serialized_index)


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    assignments: tuple[VectorAssignment, ...],
    dimension: int,
    index_path: Path,
) -> None:
    """校验 manifest 与 SQLite 映射、索引文件是否属于同一批次。"""
    if manifest.get("index_schema_version") != _INDEX_SCHEMA_VERSION:
        raise FaissIndexError("Unsupported FAISS manifest schema version.")
    if manifest.get("backend_id") != "faiss_flat_ip":
        raise FaissIndexError("FAISS manifest backend does not match the configured backend.")
    if manifest.get("index_type") != "IndexIDMap2(IndexFlatIP)":
        raise FaissIndexError("FAISS manifest records an unexpected index type.")
    if manifest.get("dimension") != dimension or manifest.get("vector_count") != len(assignments):
        raise FaissIndexError("FAISS manifest dimension or vector count is inconsistent.")
    if manifest.get("assignment_fingerprint_sha256") != _assignment_fingerprint(assignments):
        raise FaissIndexError("FAISS manifest does not match SQLite vector_id assignments.")
    if manifest.get("index_sha256") != _file_sha256(index_path):
        raise FaissIndexError("FAISS manifest checksum does not match the index file.")


def _load_manifest(path: Path) -> dict[str, Any]:
    """安全读取正式索引清单，拒绝损坏或非对象 JSON。"""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FaissIndexError(f"Cannot read FAISS manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise FaissIndexError("FAISS manifest must be a JSON object.")
    return manifest


def _load_journal(path: Path) -> dict[str, Any]:
    """读取并做最小结构校验，避免恢复流程依据随意的 JSON 执行文件替换。"""
    journal = _load_manifest(path)
    required = {
        "journal_schema_version",
        "operation_id",
        "phase",
        "index_path",
        "manifest_path",
        "candidate_index_path",
        "assignments",
        "manifest",
    }
    if journal.get("journal_schema_version") != _JOURNAL_SCHEMA_VERSION or not required.issubset(journal):
        raise FaissIndexError("Pending FAISS journal has an unsupported structure.")
    if not isinstance(journal["assignments"], list) or not isinstance(journal["manifest"], dict):
        raise FaissIndexError("Pending FAISS journal has invalid assignments or manifest.")
    return journal


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """同目录临时文件 + os.replace 写 JSON，避免留下半截 manifest/journal。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _assignment_fingerprint(assignments: tuple[VectorAssignment, ...]) -> str:
    """将正式 ID、chunk 身份与 embedding 输入快照绑定为一个稳定指纹。"""
    digest = hashlib.sha256()
    for assignment in sorted(assignments, key=lambda item: item.vector_id):
        digest.update(str(assignment.vector_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(assignment.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(assignment.embedding_text_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """流式计算 FAISS 文件哈希，不将索引二进制整体复制到内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_faiss() -> Any:
    """延迟导入 FAISS，使不涉及索引的配置/解析测试不依赖其二进制扩展。"""
    try:
        import faiss
    except ImportError as error:
        raise FaissIndexError("faiss-cpu is required for the faiss_flat_ip backend.") from error
    return faiss
