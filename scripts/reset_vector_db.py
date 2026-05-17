"""
reset_vector_db.py
==================
Vector DB / file_registry / processed outputs reset utility.
Original files under data/raw are never deleted.

Usage:
    python scripts/reset_vector_db.py
    python scripts/reset_vector_db.py --yes
    python scripts/reset_vector_db.py --mode full --yes
    python scripts/reset_vector_db.py --mode full --keep-normalized --yes
    python scripts/reset_vector_db.py --mode full --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import Path as _Path
from typing import Callable, Iterable

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.config import settings
from src.logger import get_logger
from src.storage.file_registry import FileRegistry
from src.storage.vector_store import VectorStore

log = get_logger(__name__)


@dataclass(frozen=True)
class ResetPaths:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    chroma_db_dir: Path
    registry_path: Path
    chunks_dir: Path
    documents_dir: Path
    summaries_dir: Path
    normalized_cache_dir: Path
    normalized_json_dir: Path
    normalized_markdown_dir: Path


@dataclass
class ResetResult:
    mode: str
    dry_run: bool
    keep_normalized: bool
    actions: list[str] = field(default_factory=list)
    cleanup_targets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed_cleanup_paths: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


RemovePath = Callable[[Path], None]
ResetAction = Callable[[], None]


def paths_from_settings() -> ResetPaths:
    processed_dir = Path(
        getattr(settings, "processed_data_dir", settings.project_root / "data" / "processed")
    )
    normalized_dir = processed_dir / "normalized"
    return ResetPaths(
        project_root=Path(settings.project_root),
        raw_dir=Path(settings.raw_data_dir),
        processed_dir=processed_dir,
        chroma_db_dir=Path(settings.chroma_db_dir),
        registry_path=Path(settings.registry_path),
        chunks_dir=Path(settings.chunks_dir),
        documents_dir=Path(settings.documents_dir),
        summaries_dir=processed_dir / "summaries",
        normalized_cache_dir=normalized_dir / "cache",
        normalized_json_dir=normalized_dir / "json",
        normalized_markdown_dir=normalized_dir / "markdown",
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_cleanup_target(path: Path, *, paths: ResetPaths) -> None:
    resolved = path.resolve()
    project_root = paths.project_root.resolve()
    raw_dir = paths.raw_dir.resolve()

    if not _is_relative_to(resolved, project_root):
        raise ValueError(f"Refusing to clean path outside project root: {resolved}")
    if resolved == raw_dir or _is_relative_to(resolved, raw_dir):
        raise ValueError(f"Refusing to clean data/raw path: {resolved}")


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return 1
    return sum(1 for child in path.rglob("*") if child.is_file() or child.is_symlink())


def collect_reset_stats(paths: ResetPaths, result: ResetResult) -> dict[str, object]:
    return {
        "raw_files_count": _count_files(paths.raw_dir),
        "processed_files_count": _count_files(paths.processed_dir),
        "normalized_json_count": _count_files(paths.normalized_json_dir),
        "normalized_markdown_count": _count_files(paths.normalized_markdown_dir),
        "normalized_cache_files_count": _count_files(paths.normalized_cache_dir),
        "chroma_db_exists": paths.chroma_db_dir.exists(),
        "registry_exists": paths.registry_path.exists(),
        "warning_count": result.warning_count,
        "failed_cleanup_paths": list(result.failed_cleanup_paths),
    }


def _default_remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _cleanup_path(
    path: Path,
    *,
    dry_run: bool,
    result: ResetResult,
    remove_path: RemovePath,
) -> None:
    if not path.exists() and not path.is_symlink():
        return

    result.cleanup_targets.append(str(path))
    if dry_run:
        result.actions.append(f"DRY-RUN · would remove {path}")
        return

    try:
        remove_path(path)
        result.actions.append(f"OK · removed {path}")
    except Exception as exc:  # noqa: BLE001
        message = (
            f"WARNING · failed to remove {path}: {exc}. "
            "On Windows this can happen with locked or broken residual files; "
            "review the path and continue development if it is not tracked and not under data/raw."
        )
        result.warnings.append(message)
        result.failed_cleanup_paths.append(str(path))
        log.warning(message)


def _cleanup_directory_contents(
    directory: Path,
    *,
    dry_run: bool,
    result: ResetResult,
    remove_path: RemovePath,
) -> None:
    if not directory.exists():
        return
    if directory.is_file() or directory.is_symlink():
        _cleanup_path(directory, dry_run=dry_run, result=result, remove_path=remove_path)
        return

    for child in sorted(directory.iterdir(), key=lambda p: str(p).lower()):
        _cleanup_path(child, dry_run=dry_run, result=result, remove_path=remove_path)


def _processed_root_residual_files(processed_dir: Path) -> Iterable[Path]:
    if not processed_dir.exists() or not processed_dir.is_dir():
        return []
    return [
        child
        for child in processed_dir.iterdir()
        if child.is_file() or child.is_symlink()
    ]


def cleanup_targets_for_mode(
    *,
    mode: str,
    paths: ResetPaths,
    keep_normalized: bool,
    keep_processed: bool,
) -> list[Path]:
    if mode == "soft" or keep_processed:
        return []
    if mode != "full":
        raise ValueError(f"Unsupported reset mode: {mode}")

    targets = [
        paths.chunks_dir,
        paths.documents_dir,
        paths.summaries_dir,
        *_processed_root_residual_files(paths.processed_dir),
    ]
    if not keep_normalized:
        targets.extend([
            paths.normalized_cache_dir,
            paths.normalized_json_dir,
            paths.normalized_markdown_dir,
        ])
    return targets


def reset_project(
    *,
    paths: ResetPaths,
    mode: str = "soft",
    dry_run: bool = False,
    keep_registry: bool = False,
    keep_processed: bool = False,
    keep_normalized: bool = False,
    reset_chroma: ResetAction | None = None,
    reset_registry: ResetAction | None = None,
    remove_path: RemovePath = _default_remove_path,
) -> ResetResult:
    result = ResetResult(mode=mode, dry_run=dry_run, keep_normalized=keep_normalized)
    targets = cleanup_targets_for_mode(
        mode=mode,
        paths=paths,
        keep_normalized=keep_normalized,
        keep_processed=keep_processed,
    )
    for target in targets:
        _validate_cleanup_target(target, paths=paths)

    if dry_run:
        result.actions.append("DRY-RUN · would reset ChromaDB collection")
    else:
        try:
            (reset_chroma or (lambda: VectorStore().reset_db()))()
            result.actions.append("OK · ChromaDB collection reset")
        except Exception as exc:  # noqa: BLE001
            message = f"WARNING · ChromaDB reset failed: {exc}"
            result.warnings.append(message)
            log.warning(message)

    if not keep_registry:
        if dry_run:
            result.actions.append(f"DRY-RUN · would reset file registry {paths.registry_path}")
        else:
            try:
                (reset_registry or (lambda: FileRegistry().reset()))()
                result.actions.append("OK · file registry reset")
            except Exception as exc:  # noqa: BLE001
                message = f"WARNING · file registry reset failed: {exc}"
                result.warnings.append(message)
                log.warning(message)

    for target in targets:
        if target == paths.chunks_dir or target == paths.documents_dir or target == paths.summaries_dir:
            _cleanup_directory_contents(
                target, dry_run=dry_run, result=result, remove_path=remove_path
            )
        elif target in {
            paths.normalized_cache_dir,
            paths.normalized_json_dir,
            paths.normalized_markdown_dir,
        }:
            _cleanup_directory_contents(
                target, dry_run=dry_run, result=result, remove_path=remove_path
            )
        else:
            _cleanup_path(target, dry_run=dry_run, result=result, remove_path=remove_path)

    result.stats = collect_reset_stats(paths, result)
    return result


def _print_plan(args: argparse.Namespace, paths: ResetPaths) -> None:
    print("== reset 대상 ==")
    print(f"- Mode               : {args.mode}")
    print(f"- ChromaDB           : {paths.chroma_db_dir}")
    if not args.keep_registry:
        print(f"- File Registry      : {paths.registry_path}")
    if args.mode == "full" and not args.keep_processed:
        print(f"- Processed chunks   : {paths.chunks_dir}")
        print(f"- Processed docs     : {paths.documents_dir}")
        print(f"- Processed summaries: {paths.summaries_dir}")
        print(f"- Processed root files: {paths.processed_dir}")
        if args.keep_normalized:
            print("- Normalized outputs : keep")
        else:
            print(f"- Normalized cache   : {paths.normalized_cache_dir}")
            print(f"- Normalized JSON    : {paths.normalized_json_dir}")
            print(f"- Normalized Markdown: {paths.normalized_markdown_dir}")
    print(f"- Raw data protected : {paths.raw_dir}")


def _print_result(result: ResetResult) -> None:
    print("\n== 실행 결과 ==")
    for action in result.actions:
        print(action)

    if result.warnings:
        print("\n== warnings ==")
        for warning in result.warnings:
            print(warning)

    print("\n== reset 후 상태 ==")
    for key, value in result.stats.items():
        print(f"- {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["soft", "full"],
        default="soft",
        help="soft: ChromaDB/file registry reset only. full: processed outputs cleanup too.",
    )
    parser.add_argument("--full", action="store_true", help="Shortcut for --mode full")
    parser.add_argument("--yes", action="store_true", help="확인 프롬프트 없이 실행")
    parser.add_argument("--dry-run", action="store_true", help="삭제 없이 대상과 작업만 출력")
    parser.add_argument("--keep-registry", action="store_true", help="indexed_files.json 은 유지")
    parser.add_argument(
        "--keep-processed",
        action="store_true",
        help="full mode 에서 data/processed 산출물을 유지",
    )
    parser.add_argument(
        "--keep-normalized",
        action="store_true",
        help="full mode 에서 processed/normalized 산출물을 유지",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.full:
        args.mode = "full"

    paths = paths_from_settings()
    _print_plan(args, paths)

    if not args.yes and not args.dry_run:
        ans = input("정말 초기화하시겠습니까? (y/N): ").strip().lower()
        if ans != "y":
            print("취소.")
            return 0

    try:
        result = reset_project(
            paths=paths,
            mode=args.mode,
            dry_run=args.dry_run,
            keep_registry=args.keep_registry,
            keep_processed=args.keep_processed,
            keep_normalized=args.keep_normalized,
        )
    except ValueError as exc:
        print(f"중단: {exc}")
        return 2

    _print_result(result)
    print("완료.")
    return 0 if not result.failed_cleanup_paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
