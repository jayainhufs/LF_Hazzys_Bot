from __future__ import annotations

from pathlib import Path

import pytest

from scripts.reset_vector_db import ResetPaths, reset_project


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _paths(tmp_path: Path) -> ResetPaths:
    project = tmp_path / "project"
    raw_dir = project / "data" / "raw"
    processed_dir = project / "data" / "processed"
    normalized_dir = processed_dir / "normalized"
    return ResetPaths(
        project_root=project,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chroma_db_dir=project / "storage" / "chroma_db",
        registry_path=project / "storage" / "indexed_files.json",
        chunks_dir=processed_dir / "chunks",
        documents_dir=processed_dir / "documents",
        summaries_dir=processed_dir / "summaries",
        normalized_cache_dir=normalized_dir / "cache",
        normalized_json_dir=normalized_dir / "json",
        normalized_markdown_dir=normalized_dir / "markdown",
    )


def _seed_outputs(paths: ResetPaths) -> None:
    _touch(paths.raw_dir / "guide.docx")
    _touch(paths.chunks_dir / "chunk.json")
    _touch(paths.documents_dir / "doc.json")
    _touch(paths.summaries_dir / "excel" / "summary.json")
    _touch(paths.normalized_cache_dir / "cache.json")
    _touch(paths.normalized_json_dir / "normalized.json")
    _touch(paths.normalized_markdown_dir / "normalized.md")
    _touch(paths.processed_dir / "MP7AUH.DOCX")
    _touch(paths.registry_path)
    _touch(paths.chroma_db_dir / "placeholder")


def test_full_reset_clears_processed_outputs_and_preserves_raw(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_outputs(paths)

    result = reset_project(
        paths=paths,
        mode="full",
        reset_chroma=lambda: None,
        reset_registry=lambda: None,
    )

    assert result.warning_count == 0
    assert (paths.raw_dir / "guide.docx").exists()
    assert not any(paths.chunks_dir.rglob("*"))
    assert not any(paths.documents_dir.rglob("*"))
    assert not any(paths.summaries_dir.rglob("*"))
    assert not any(paths.normalized_cache_dir.rglob("*"))
    assert not any(paths.normalized_json_dir.rglob("*"))
    assert not any(paths.normalized_markdown_dir.rglob("*"))
    assert not (paths.processed_dir / "MP7AUH.DOCX").exists()


def test_full_reset_keep_normalized_preserves_normalized_outputs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_outputs(paths)

    reset_project(
        paths=paths,
        mode="full",
        keep_normalized=True,
        reset_chroma=lambda: None,
        reset_registry=lambda: None,
    )

    assert not any(paths.chunks_dir.rglob("*"))
    assert not any(paths.documents_dir.rglob("*"))
    assert not any(paths.summaries_dir.rglob("*"))
    assert (paths.normalized_cache_dir / "cache.json").exists()
    assert (paths.normalized_json_dir / "normalized.json").exists()
    assert (paths.normalized_markdown_dir / "normalized.md").exists()


def test_dry_run_does_not_delete_files_or_reset_backends(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_outputs(paths)
    calls: list[str] = []

    result = reset_project(
        paths=paths,
        mode="full",
        dry_run=True,
        reset_chroma=lambda: calls.append("chroma"),
        reset_registry=lambda: calls.append("registry"),
    )

    assert calls == []
    assert result.dry_run is True
    assert (paths.chunks_dir / "chunk.json").exists()
    assert (paths.normalized_json_dir / "normalized.json").exists()
    assert (paths.processed_dir / "MP7AUH.DOCX").exists()


def test_soft_reset_only_resets_chroma_and_registry(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_outputs(paths)
    calls: list[str] = []

    result = reset_project(
        paths=paths,
        mode="soft",
        reset_chroma=lambda: calls.append("chroma"),
        reset_registry=lambda: calls.append("registry"),
    )

    assert calls == ["chroma", "registry"]
    assert result.cleanup_targets == []
    assert (paths.chunks_dir / "chunk.json").exists()
    assert (paths.normalized_json_dir / "normalized.json").exists()


def test_cleanup_failure_is_recorded_as_warning_and_continues(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_outputs(paths)
    locked_file = paths.chunks_dir / "chunk.json"

    def failing_remove(path: Path) -> None:
        if path == locked_file:
            raise PermissionError("locked")
        if path.is_dir():
            for child in path.iterdir():
                failing_remove(child)
            path.rmdir()
        else:
            path.unlink()

    result = reset_project(
        paths=paths,
        mode="full",
        reset_chroma=lambda: None,
        reset_registry=lambda: None,
        remove_path=failing_remove,
    )

    assert result.warning_count == 1
    assert str(locked_file) in result.failed_cleanup_paths
    assert locked_file.exists()
    assert not (paths.normalized_json_dir / "normalized.json").exists()


def test_raw_path_cleanup_target_is_refused(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    raw_file = _touch(paths.raw_dir / "guide.docx")
    unsafe_paths = ResetPaths(
        project_root=paths.project_root,
        raw_dir=paths.raw_dir,
        processed_dir=paths.processed_dir,
        chroma_db_dir=paths.chroma_db_dir,
        registry_path=paths.registry_path,
        chunks_dir=paths.raw_dir / "bad_chunks",
        documents_dir=paths.documents_dir,
        summaries_dir=paths.summaries_dir,
        normalized_cache_dir=paths.normalized_cache_dir,
        normalized_json_dir=paths.normalized_json_dir,
        normalized_markdown_dir=paths.normalized_markdown_dir,
    )

    with pytest.raises(ValueError, match="data/raw"):
        reset_project(
            paths=unsafe_paths,
            mode="full",
            reset_chroma=lambda: None,
            reset_registry=lambda: None,
        )

    assert raw_file.exists()
