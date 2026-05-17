"""
1_document_upload.py
====================
4가지 업로드 카테고리 + 기타 카테고리 별로 파일을 drag-and-drop 으로 업로드한다.
업로드 후 색인은 '문서 색인' 페이지에서 진행한다.
"""
from __future__ import annotations

import app._bootstrap  # noqa: F401

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.upload_box import (
    render_category_selector,
    render_uploader,
    save_uploaded_files,
)
from src.config import settings
from src.utils.path_utils import iter_files

st.set_page_config(page_title="문서 업로드", layout="wide")
render_sidebar()

st.title("문서 업로드")
st.caption(
    "업로드된 파일은 카테고리별 `data/raw/<카테고리>/` 폴더에 저장됩니다. "
    "이후 색인 페이지에서 색인하세요."
)

uploaded_category = render_category_selector()
target_dir = settings.category_to_dir(uploaded_category)
st.info(f"저장 폴더: `{target_dir.relative_to(settings.project_root)}`")

uploads = render_uploader(uploaded_category)

if uploads:
    if st.button(f"{len(uploads)}개 파일 저장하기", type="primary"):
        saved = save_uploaded_files(uploads, uploaded_category)
        if saved:
            st.success(f"{len(saved)}개 파일을 저장했습니다.")
            for p, _status in saved:
                st.write(f"- [OK] {p.relative_to(settings.project_root)}")
            st.warning(
                "아직 색인되지 않았습니다. 좌측 '문서 색인' 페이지에서 색인을 실행하세요."
            )

st.divider()
st.subheader("카테고리 폴더에 이미 있는 파일")
for label, sub in settings.category_dirs.items():
    folder = settings.raw_data_dir / sub
    files = list(iter_files(folder, exts={".xlsx", ".xlsm", ".docx", ".txt", ".md"}))
    with st.expander(f"{label} · {sub} ({len(files)}개)", expanded=False):
        if not files:
            st.caption("(비어있음)")
            continue
        for p in files:
            st.write(f"- {p.name}  ·  {p.stat().st_size//1024} KB")
