# -*- coding: utf-8 -*-
"""策略族帮助：折叠一句话 + 页内阅读/下载说明文档。"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def render_strategy_family_help(st: Any, *, key_prefix: str = "strategy_help") -> None:
    """在策略族下拉附近展示轻量帮助。

    - 折叠区：各族一句话
    - 可展开阅读 docs/策略族说明.md
    - 提供下载按钮与本地路径
    """
    from src.strategies.ashare_strategy_runner import (
        STRATEGY_HELP_DOC_REL,
        format_strategy_family_oneliners_md,
        get_strategy_help_doc_path,
    )

    doc_path = get_strategy_help_doc_path()
    with st.expander("策略族说明（一句话 / 详细文档）", expanded=False):
        st.markdown(format_strategy_family_oneliners_md())
        st.caption(
            "完整说明见项目文件：`%s`（绝对路径：`%s`）"
            % (STRATEGY_HELP_DOC_REL, doc_path)
        )
        if doc_path.is_file():
            try:
                md_text = doc_path.read_text(encoding="utf-8")
            except OSError as e:
                st.warning("无法读取说明文档：%s" % e)
                md_text = ""
            if md_text:
                with st.expander("在页面内查看详细说明", expanded=False):
                    st.markdown(md_text)
                st.download_button(
                    label="下载《策略族说明》Markdown",
                    data=md_text.encode("utf-8"),
                    file_name="策略族说明.md",
                    mime="text/markdown; charset=utf-8",
                    key="%s_download_md" % key_prefix,
                )
            # file:/// 链接在部分浏览器可用；同时给出路径便于资源管理器打开
            file_url = doc_path.resolve().as_uri()
            st.markdown("本地打开：[策略族说明.md](%s)" % file_url)
        else:
            st.warning("未找到说明文档：%s" % doc_path)
