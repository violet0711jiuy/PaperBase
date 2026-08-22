"""PaperBase Streamlit 前端入口。

本文件负责完成三件事：

1. 让 Python 能找到项目根目录下的 ``paperbase`` 后端包；
2. 初始化 Streamlit 页面和会话状态；
3. 根据全局导航的选择渲染 Knowledge Base 或 Paper Workspace。

当前版本接入 Knowledge Base 多会话问答、Paper Overview、Explain Section 和
Paper Workspace 的 Ask This Paper；各页面仍通过 Service Layer 调用后端能力。
"""

# 允许使用 ``from __future__`` 的类型标注语法，同时不影响旧版本 Python 的解析方式。
from __future__ import annotations

# ``Path`` 用来稳定地计算当前文件、app 目录和项目根目录的位置。
from pathlib import Path
# ``sys`` 用来补充 Python 的模块搜索路径。
import sys

# Streamlit 启动脚本时通常只保证 ``app/`` 在 ``sys.path`` 中。
# 但 ``paperbase/`` 与 ``app/`` 是同级目录，所以这里需要把项目根目录补进去。
# 这样无论用户从哪个当前工作目录启动，都可以导入 PaperBase 后端。
# 当前入口文件所在的绝对目录，也就是项目中的 ``app/`` 目录。
_APP_DIR = Path(__file__).resolve().parent
# ``app/`` 的父目录就是 PaperBase 项目根目录。
_PROJECT_ROOT = _APP_DIR.parent
# 依次确保项目根目录和 app 目录都存在于模块搜索路径中。
for _path in (_PROJECT_ROOT, _APP_DIR):
    # 只有路径尚未加入时才插入，避免重复污染 ``sys.path``。
    if str(_path) not in sys.path:
        # 插入到列表最前面，让当前项目代码优先于同名的外部包。
        sys.path.insert(0, str(_path))

# Streamlit 是页面渲染、Tab 和表单控件的基础库。
import streamlit as st

# 入口脚本直接运行时，``app/`` 是最容易被 Streamlit 找到的导入根目录。
# 因此这里使用不带 ``app.`` 前缀的导入形式，兼容 ``streamlit run app/app.py``。
# 第一栏的全局导航。
from components.sidebar import render_navigation_pane
# 统一的三栏 App Shell。
from components.layout import render_app_shell
# 两个页面模块分别负责 Knowledge Base 和 Paper Workspace 的骨架。
from pages import knowledge_base, paper_workspace
# Service 是页面访问后端能力的唯一适配层。
from services.paperbase_service import PaperBaseService
# 状态模块提供当前页面常量和 Session State 初始化函数。
from state import KNOWLEDGE_BASE_PAGE, initialize_session_state


def main() -> None:
    """启动整个 Streamlit 页面，并按当前导航项选择页面。"""
    # 设置浏览器标题、页面图标和宽屏布局。
    st.set_page_config(page_title="PaperBase", page_icon="📚", layout="wide")
    # 只补齐缺失状态，不覆盖用户在 rerun 中已经选择的状态。
    initialize_session_state(st.session_state)
    # 创建一个轻量 Service 实例，页面通过它读取论文和会话信息。
    service = PaperBaseService()
    # 所有页面都先创建同一套 Global Nav / Context / Main 三栏。
    navigation_column, context_column, main_column = render_app_shell()
    with navigation_column:
        render_navigation_pane()

    # 只切换第二栏和第三栏的内容，避免页面之间互相创建不同布局。
    current_page = st.session_state.current_page
    if current_page == KNOWLEDGE_BASE_PAGE:
        with context_column:
            knowledge_base.render_context_panel(service)
        # 两个页面共用同一张主内容卡片。页面模块只提供业务内容，避免它们
        # 各自用 CSS 猜测右栏边界，从而造成布局和滚动规则互相污染。
        with main_column, st.container(key="pb-main-card"):
            knowledge_base.render_main_panel(service)
    else:
        with context_column:
            paper_workspace.render_context_panel(service)
        with main_column, st.container(key="pb-main-card"):
            paper_workspace.render_main_panel(service)

if __name__ == "__main__":
    # 只有直接执行此文件时才启动；被测试工具导入时不会自动启动页面。
    main()
