# PaperBase Streamlit 前端阅读指南

这份说明只讲当前 `app/` 目录里最需要看懂的部分，方便以后自己调整 UI。

## 1. Streamlit 的基本运行方式

Streamlit 和 React/Vue 不一样：每当你点击按钮、切换选择项、提交输入框时，当前 Python 脚本会从上到下重新执行一遍（rerun）。

因此页面状态不能只放在普通变量里。需要跨 rerun 保留的状态使用：

```python
st.session_state.xxx
```

PaperBase 当前主要保存：

- `current_page`：知识库 / 论文工作区
- `active_workspace_id`：当前 staging 论文
- `selected_section_id`：当前选中的章节
- `kb_conversation_id`：当前知识库会话
- `paper_conversation_id`：当前论文会话

完整聊天历史并不保存在 `session_state`，而是由后端 `conversations.sqlite3` 持久化。

## 2. app.py：前端入口

`app/app.py` 是 Streamlit 入口。

它主要做：

```text
初始化页面
→ 初始化 session_state
→ 创建 PaperBaseService
→ 建立三栏 App Shell
→ 第一栏渲染全局导航
→ 第二栏渲染当前工作区控制面板
→ 第三栏渲染主内容
```

启动方式仍然是：

```bash
streamlit run app/app.py
```

## 3. components/layout.py：三栏骨架

这里通过：

```python
st.columns([0.15, 0.22, 0.63], gap=None)
```

建立三栏：

- 15%：全局导航
- 22%：当前工作区上下文
- 63%：主内容区

`gap=None` 的目的，是让第一栏和第二栏真正挨在一起，而不是出现 Streamlit 默认的大间距。

## 4. components/styles.py：UI 样式中心

本次把原来塞在 `layout.py` 里的大段 CSS 独立到了：

```text
app/components/styles.py
```

以后想改：

- 背景色
- 三栏颜色
- 按钮颜色
- 聊天气泡
- 字号
- 圆角
- Evidence 卡片
- Overview 卡片
- 滚动区域高度

优先在这里改，不要在各页面到处写 CSS。

几个最重要的 CSS 变量：

```css
--pb-bg
--pb-nav-bg
--pb-context-bg
--pb-blue
--pb-chat-scroll-height
--pb-paper-chat-scroll-height
--pb-paper-tab-height
```

## 5. 为什么页面不整体滚动

全局样式把应用固定在浏览器视口：

```css
height: 100dvh;
overflow: hidden;
```

然后单独给聊天记录区设置固定高度和：

```css
overflow-y: auto;
```

所以：

```text
整个页面固定
├─ 左栏固定
├─ 中栏自己需要时滚动
└─ 主聊天区
   ├─ Header 固定
   ├─ 消息区滚动
   └─ 输入框固定
```

这比让整个 Streamlit 页面上下滑更像正常聊天软件。

## 6. pages/knowledge_base.py：知识库聊天页

它分两部分：

```python
render_context_panel()
```

负责第二栏：

- 新建会话
- 会话列表
- 已入库论文

```python
render_main_panel()
```

负责第三栏：

- 当前会话 Header
- 聊天历史
- Evidence
- 底部输入框

消息采用左右布局：

```text
用户 → 右侧
PaperBase → 左侧
```

头像现在使用内联 SVG，而不是 emoji，因此 Windows / macOS / 浏览器之间更稳定。

## 7. pages/paper_workspace.py：论文工作区

第二栏负责：

- 上传论文
- 最近 staging 工作区
- 当前论文状态

第三栏负责：

- 论文标题
- 论文概览
- 解释章节
- 询问本文

三个功能仍然使用 Streamlit Tabs：

```python
st.tabs(...)
```

论文概览、Explain 和 Ask 都在主卡片内部滚动，不再把整个网页撑长。

## 8. Service Layer 为什么存在

页面不会自己打开 SQLite、FAISS 或 staging JSON。

UI 统一调用：

```text
app/services/paperbase_service.py
```

例如：

```python
service.list_workspaces()
service.ask_knowledge_base(...)
service.get_paper_overview(...)
```

这层的作用是：

```text
Streamlit UI
    ↓
PaperBaseService
    ↓
paperbase 后端
```

以后就算从 Streamlit 换成 FastAPI + React，RAG 后端也不用重写。

## 9. 最常用的 Streamlit 控件

```python
st.button()
```
按钮。点击后触发 rerun。

```python
st.columns()
```
横向布局。

```python
st.container()
```
逻辑容器。加 `key` 后可以用 CSS 精确控制样式。

```python
st.container(height=650)
```
固定高度的内部滚动容器，本项目用于聊天区和 Explain 阅读区。

```python
st.expander()
```
折叠区域，用于 Evidence / 来源。

```python
st.tabs()
```
Tab 页面，用于论文概览 / 解释章节 / 询问本文。

```python
st.form()
st.text_input()
st.form_submit_button()
```
组成底部聊天输入区。

## 10. 以后想改 UI 时怎么判断改哪个文件

- 三栏宽度、全局配色、字体、圆角 → `components/styles.py`
- 全局导航 → `components/sidebar.py`
- 知识库会话和聊天排版 → `pages/knowledge_base.py`
- 论文工作区结构 → `pages/paper_workspace.py`
- Overview 每个字段怎么显示 → `components/overview.py`
- Explain 目录 → `components/section_tree.py`
- Evidence 展开方式 → `pages/knowledge_base.py` / `components/source_evidence.py`
- 后端数据怎么取 → `services/paperbase_service.py`

原则：**样式问题先找 `styles.py`，业务结构问题再找 page/component。**
