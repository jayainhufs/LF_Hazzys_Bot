"""
encoding_utils.py
=================
텍스트 파일 인코딩 추정/디코딩 유틸.
한글 환경에서는 UTF-8, UTF-8-SIG, CP949, EUC-KR 을 순차적으로 시도한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

DEFAULT_ENCODINGS: Tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "cp949",
    "euc-kr",
    "latin-1",
)


def read_text_safely(path: Path, encodings: List[str] | None = None) -> Tuple[str, str]:
    """
    파일을 안전하게 텍스트로 읽는다.

    Returns
    -------
    (content, used_encoding)

    Raises
    ------
    UnicodeDecodeError
        모든 인코딩 시도가 실패한 경우.
    """
    encodings = list(encodings) if encodings else list(DEFAULT_ENCODINGS)
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc), enc
        except (UnicodeDecodeError, UnicodeError) as e:  # noqa: PERF203
            last_err = e
            continue
        except Exception as e:  # 권한/IO 등 다른 예외는 즉시 raise
            raise e
    if last_err:
        raise last_err
    raise UnicodeDecodeError("auto", b"", 0, 1, f"모든 인코딩 디코딩 실패: {path}")


def decode_bytes_safely(data: bytes, encodings: List[str] | None = None) -> Tuple[str, str]:
    """bytes 를 안전하게 디코딩."""
    encodings = list(encodings) if encodings else list(DEFAULT_ENCODINGS)
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError as e:  # noqa: PERF203
            last_err = e
            continue
    if last_err:
        raise last_err
    return data.decode("utf-8", errors="ignore"), "utf-8"
