"""
path_utils.py
=============
Mac / Windows 양쪽에서 안전하게 동작하는 경로 유틸.
모든 경로는 반드시 pathlib.Path 로 다룬다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# 파일명에 들어가면 OS별로 문제가 될 수 있는 문자
_FORBIDDEN_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def safe_filename(name: str, max_length: int = 180) -> str:
    """OS-safe 파일명으로 정규화. 한글은 유지한다."""
    name = name.strip()
    name = _FORBIDDEN_FILENAME_CHARS.sub("_", name)
    name = re.sub(r"\s+", " ", name)
    if len(name) > max_length:
        # 확장자 보존
        if "." in name:
            base, ext = name.rsplit(".", 1)
            name = base[: max_length - len(ext) - 1] + "." + ext
        else:
            name = name[:max_length]
    return name or "untitled"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_files(root: Path, exts: Iterable[str] | None = None) -> Iterable[Path]:
    """root 하위의 파일들을 재귀적으로 순회. ext 는 '.xlsx' 형식."""
    exts_lower = {e.lower() for e in exts} if exts else None
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.endswith(".gitkeep"):
            continue
        if exts_lower and p.suffix.lower() not in exts_lower:
            continue
        yield p


def relative_to_project(path: Path, project_root: Path) -> str:
    """프로젝트 루트 기준 상대 경로 문자열을 반환. 실패 시 그냥 str(path)."""
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)
