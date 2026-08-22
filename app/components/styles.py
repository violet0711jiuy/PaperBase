"""PaperBase Streamlit 的统一视觉主题。

这里集中维护三栏工作台、聊天气泡、论文工作区、Overview/Explain 等页面样式。
页面模块只负责业务结构，不再各自散落大段 CSS。
"""

from __future__ import annotations

import streamlit as st


def apply_global_styles() -> None:
    """注入 PaperBase 全局 CSS。

    核心约束：
    - 整个应用固定在浏览器视口内，避免整页滚动；
    - 第一、第二栏紧邻，第三栏占主要宽度；
    - Knowledge Base / Ask This Paper 只让消息列表滚动；
    - 论文 Overview / Explain 使用主内容区内部滚动。
    """
    st.markdown(
        """
<style>
:root {
  --pb-bg: #f4f7fb;
  --pb-nav-bg: #f8faff;
  --pb-context-bg: #f8fafd;
  --pb-card: #ffffff;
  --pb-card-soft: #f8faff;
  --pb-border: #e2e8f1;
  --pb-border-strong: #d4deeb;
  --pb-text: #18243a;
  --pb-text-2: #3f4f68;
  --pb-muted: #7b8aa0;
  --pb-blue: #3f63bf;
  --pb-blue-2: #3155ae;
  --pb-blue-soft: #eef4ff;
  --pb-user-bubble: #e8f1ff;
  --pb-shadow: 0 7px 22px rgba(35, 55, 88, 0.055);
  --pb-shell-height: 100dvh;
  --pb-chat-scroll-height: calc(100dvh - 210px);
  --pb-paper-chat-scroll-height: calc(100dvh - 390px);
  --pb-paper-tab-height: calc(100dvh - 175px);
  --pb-explain-height: calc(100dvh - 190px);
}

html, body, .stApp, button, input, textarea, [data-testid="stMarkdownContainer"] {
  font-family: "Inter", "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif !important;
}

/* ---------- Streamlit 外壳：固定视口，页面本身不滚 ---------- */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  height: 100dvh !important;
  min-height: 100dvh !important;
  overflow: hidden !important;
  background: var(--pb-bg) !important;
  color: var(--pb-text);
}

header[data-testid="stHeader"] {
  height: 0 !important;
  min-height: 0 !important;
  background: transparent !important;
}

#MainMenu, footer { visibility: hidden; }

.block-container {
  max-width: none !important;
  height: 100dvh !important;
  min-height: 100dvh !important;
  overflow: hidden !important;
  padding: 0 !important;
}

/* Streamlit 默认 widget 统一圆角 */
.stButton button,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
div[data-testid="stTextInput"] input {
  border-radius: 10px !important;
}

.stButton button {
  min-height: 2.55rem;
  font-weight: 600;
}

.stButton button[kind="primary"] {
  background: linear-gradient(180deg, #4a6bd0 0%, #3b5fbd 100%);
  border-color: #3f63bf;
  color: #fff;
  box-shadow: 0 4px 10px rgba(63, 99, 191, 0.16);
}

.stButton button[kind="primary"]:hover {
  background: #3458b5;
  border-color: #3458b5;
}

.stButton button[kind="secondary"] {
  color: var(--pb-text-2);
}

/* ---------- 三栏 App Shell ---------- */
.st-key-pb-global-nav,
.st-key-pb-context-panel,
.st-key-pb-main-panel {
  box-sizing: border-box;
  height: var(--pb-shell-height) !important;
  min-height: var(--pb-shell-height) !important;
  max-height: var(--pb-shell-height) !important;
}

.st-key-pb-global-nav {
  background: var(--pb-nav-bg);
  border-right: 1px solid var(--pb-border);
  overflow: hidden;
  padding: 1.65rem 1.2rem 1.15rem;
}

.st-key-pb-context-panel {
  background: var(--pb-context-bg);
  border-right: 1px solid var(--pb-border);
  overflow-x: hidden;
  overflow-y: auto;
  padding: 1.65rem 1.25rem 1.1rem;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e3 transparent;
}

.st-key-pb-main-panel {
  background: var(--pb-bg);
  overflow: hidden;
  padding: 0.72rem 0.85rem;
}

.st-key-pb-main-card {
  background: var(--pb-card);
  border: 1px solid var(--pb-border);
  border-radius: 14px;
  box-shadow: var(--pb-shadow);
  box-sizing: border-box;
  height: calc(100dvh - 1.44rem) !important;
  max-height: calc(100dvh - 1.44rem) !important;
  overflow: hidden !important;
  padding: 1.05rem 1.35rem 0.8rem;
}

/* ---------- 第一栏 ---------- */
.pb-brand-row {
  align-items: center;
  display: flex;
  gap: 0.62rem;
  padding: 0.25rem 0.1rem 1.05rem;
}

.pb-brand-logo {
  align-items: center;
  display: flex;
  height: 2rem;
  justify-content: center;
  width: 2rem;
}

.pb-brand-logo svg { height: 1.85rem; width: 1.85rem; }
.pb-brand-name { color: #13203a; font-size: 1.52rem; font-weight: 780; line-height: 1.05; }
.pb-brand-subtitle { color: var(--pb-muted); font-size: 0.86rem; margin-top: 0.28rem; }
.pb-nav-section-title { color: #718096; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.07em; margin: 0.6rem 0 0.5rem; }

.st-key-pb-global-nav hr {
  border-color: var(--pb-border);
  margin: 0 0 0.75rem;
}

.st-key-pb-global-nav .stButton { margin: 0.18rem 0; }
.st-key-pb-global-nav .stButton button {
  border: 1px solid transparent;
  border-radius: 10px !important;
  font-size: 1.06rem;
  justify-content: flex-start;
  min-height: 3rem;
  padding: 0.62rem 0.82rem;
  text-align: left;
}

.st-key-pb-global-nav .stButton button > div,
.st-key-pb-global-nav .stButton button p,
.st-key-kb-conversation-list .stButton button > div,
.st-key-kb-conversation-list .stButton button p,
.st-key-paper-workspace-list .stButton button > div,
.st-key-paper-workspace-list .stButton button p,
.st-key-paper-section-tree .stButton button > div,
.st-key-paper-section-tree .stButton button p {
  justify-content: flex-start !important;
  text-align: left !important;
  width: 100%;
}

.st-key-pb-global-nav .stButton button[kind="secondary"] {
  background: transparent;
  border-color: transparent;
  color: #26364f;
}

.st-key-pb-global-nav .stButton button[kind="secondary"]:hover {
  background: #eef3fb;
  border-color: #e5ebf4;
  color: var(--pb-blue-2);
}

.st-key-pb-global-nav .stButton button[kind="primary"] {
  background: #eaf1ff;
  border-color: #dbe6fa;
  box-shadow: inset 3px 0 0 var(--pb-blue);
  color: var(--pb-blue-2);
}

/* ---------- 第二栏通用 ---------- */
.pb-pane-kicker { color: var(--pb-muted); font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em; }
.pb-pane-title { color: var(--pb-text); font-size: 1.58rem; font-weight: 760; line-height: 1.2; margin-top: 0.2rem; }
.pb-pane-subtitle { color: #6f7f96; font-size: 0.9rem; line-height: 1.5; margin-top: 0.28rem; }
.pb-context-stat { color: #52647c; font-size: 0.92rem; font-weight: 650; margin: 1rem 0 0.8rem; }
.pb-context-section-title { color: #26364f; font-size: 0.96rem; font-weight: 720; margin: 0.75rem 0 0.58rem; }
.pb-panel-divider { border-top: 1px solid var(--pb-border); margin: 1.15rem 0 0.85rem; }
.pb-muted { color: var(--pb-muted); font-size: 0.8rem; }

.st-key-kb-new-chat .stButton button,
.st-key-paper-upload-panel .stButton button[kind="primary"] {
  font-size: 0.98rem;
  min-height: 2.65rem;
}

.st-key-kb-conversation-list .stButton,
.st-key-paper-workspace-list .stButton { margin: 0.32rem 0; }

.st-key-kb-conversation-list .stButton button,
.st-key-paper-workspace-list .stButton button {
  align-items: flex-start;
  background: #fff;
  border: 1px solid #e3e9f2;
  border-radius: 10px !important;
  box-shadow: 0 1px 3px rgba(31, 51, 83, 0.025);
  color: #2b3a52;
  font-size: 0.92rem;
  justify-content: flex-start;
  line-height: 1.46;
  min-height: 4.15rem;
  padding: 0.72rem 0.78rem;
  text-align: left;
  white-space: pre-line;
}

.st-key-kb-conversation-list .stButton button[kind="primary"],
.st-key-paper-workspace-list .stButton button[kind="primary"] {
  background: #eef4ff;
  border-color: #86a5ee;
  box-shadow: inset 3px 0 0 var(--pb-blue);
  color: #20458e;
}

.st-key-kb-conversation-list .stButton button[kind="secondary"]:hover,
.st-key-paper-workspace-list .stButton button[kind="secondary"]:hover {
  background: #f5f8ff;
  border-color: #cad8f5;
  color: var(--pb-blue-2);
}

.st-key-kb-paper-library,
.st-key-paper-workspace-list { margin-top: 0.35rem; }

.st-key-kb-paper-library div[data-testid="stExpander"] {
  background: #fff;
  border-color: var(--pb-border);
  border-radius: 10px;
}

.pb-paper-item { border-bottom: 1px solid #edf1f6; padding: 0.58rem 0 0.62rem; }
.pb-paper-item:last-child { border-bottom: 0; }
.pb-paper-item-title { color: #34455e; display: -webkit-box; font-size: 0.86rem; font-weight: 620; line-height: 1.45; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
.pb-paper-item-meta { color: #8a98aa; font-size: 0.74rem; margin-top: 0.18rem; }

/* ---------- Paper Context ---------- */
.st-key-paper-upload-panel {
  background: #fff;
  border: 1px solid var(--pb-border);
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(31, 51, 83, 0.035);
  margin: 1rem 0 0.88rem;
  padding: 0.82rem;
}
.pb-upload-icon { align-items: center; color: #4a6bd0; display: flex; justify-content: center; margin-bottom: 0.38rem; }
.pb-upload-icon svg { fill: none; height: 2.1rem; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 1.6; width: 2.1rem; }
.pb-upload-panel-title { color: #24344d; font-size: 1rem; font-weight: 720; text-align: center; }
.pb-upload-panel-copy { color: var(--pb-muted); font-size: 0.78rem; line-height: 1.45; margin: 0.25rem 0 0.7rem; text-align: center; }
.pb-upload-compact { align-items: flex-start; display: flex; flex-direction: column; gap: 0.16rem; margin-bottom: 0.58rem; padding: 0.08rem 0.08rem 0.18rem; }
.pb-upload-compact strong { color: #33465f; font-size: 0.9rem; font-weight: 700; }
.pb-upload-compact span { color: #8996a8; font-size: 0.74rem; line-height: 1.4; }

.pb-workspace-status-card {
  background: #fff;
  border: 1px solid var(--pb-border);
  border-radius: 11px;
  margin-top: 0.9rem;
  padding: 0.85rem 0.9rem;
}
.pb-workspace-status-title { color: #42536c; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.45rem; }
.pb-workspace-status-row { align-items: center; color: #7c8a9d; display: flex; font-size: 0.78rem; justify-content: space-between; padding: 0.18rem 0; }
.pb-workspace-status-row strong { color: #52637b; font-weight: 650; }

/* ---------- 第三栏顶部 ---------- */
.pb-chat-topline,
.pb-paper-header-row {
  align-items: flex-start;
  display: flex;
  gap: 1rem;
  justify-content: space-between;
}
.pb-chat-title { color: #13203a; font-size: 1.38rem; font-weight: 760; line-height: 1.25; }
.pb-chat-subtitle { color: #74849a; font-size: 0.83rem; margin-top: 0.24rem; }
.pb-chat-status,
.pb-paper-status-chip {
  background: #f7faff;
  border: 1px solid #dfe7f2;
  border-radius: 999px;
  color: #64758d;
  font-size: 0.75rem;
  padding: 0.36rem 0.62rem;
  white-space: nowrap;
}
.pb-paper-status-chip.is-added { background: #f0faf4; border-color: #d7eee0; color: #4c8661; }
.pb-paper-status-chip.is-added { margin-bottom: 0.42rem; text-align: center; }
[class*="st-key-paper-workspace-actions-"] { padding-top: 0.1rem; }
[class*="st-key-paper-workspace-actions-"] .stButton + .stButton { margin-top: 0.38rem; }
[class*="st-key-paper-workspace-actions-"] .stButton button[kind="secondary"] {
  background: #fff;
  border-color: #e3d8dc;
  color: #8c555b;
}
.pb-chat-divider { border-top: 1px solid var(--pb-border); margin: 0.72rem 0 0.45rem; }

.pb-paper-header {
  border-bottom: 1px solid var(--pb-border);
  margin-bottom: 0.45rem;
  padding: 0.15rem 0.15rem 0.85rem;
}
.pb-paper-title { color: #13203a; font-size: 1.46rem; font-weight: 770; line-height: 1.3; }
.pb-paper-workspace-subtitle { color: #718198; font-size: 0.84rem; margin-top: 0.3rem; }
.pb-paper-source { color: #94a1b2; font-size: 0.74rem; margin-top: 0.22rem; }
.pb-paper-metadata { align-items: center; display: flex; gap: 1rem; justify-content: space-between; margin-top: 0.55rem; }
.pb-paper-size { color: #60718a; font-size: 0.8rem; font-weight: 630; }
.pb-paper-status { color: #8290a2; font-size: 0.8rem; }
.pb-paper-status.is-added { color: #4d8963; }

/* Tabs 更接近桌面工具栏 */
button[data-baseweb="tab"] {
  font-size: 1rem !important;
  font-weight: 650 !important;
  padding-left: 1.1rem !important;
  padding-right: 1.1rem !important;
}

/* ---------- KB / Ask 消息区：唯一滚动区 ---------- */
.st-key-kb-message-scroll,
[class*="st-key-paper-message-scroll-"] {
  background: #fff;
  border: 0;
  height: var(--pb-chat-scroll-height) !important;
  max-height: var(--pb-chat-scroll-height) !important;
  min-height: 260px !important;
  overflow-y: auto !important;
  padding: 0.7rem 0.2rem 0.55rem;
  scrollbar-width: thin;
  scrollbar-color: #cbd6e5 transparent;
}

[class*="st-key-paper-message-scroll-"] {
  height: var(--pb-paper-chat-scroll-height) !important;
  max-height: var(--pb-paper-chat-scroll-height) !important;
}

.st-key-kb-message-scroll > [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-paper-message-scroll-"] > [data-testid="stVerticalBlockBorderWrapper"] {
  height: 100% !important;
  max-height: 100% !important;
  overflow-y: auto !important;
  border: 0 !important;
}

.pb-empty-chat,
.pb-empty-panel { color: #74849a; padding: 18vh 1rem; text-align: center; }
.pb-empty-icon { color: #7d96cf; font-size: 1.5rem; }
.pb-empty-title { color: #43536b; font-size: 1rem; font-weight: 680; margin-top: 0.35rem; }
.pb-empty-copy { font-size: 0.8rem; margin-top: 0.28rem; }

/* ---------- 头像与消息 ---------- */
.pb-avatar {
  align-items: center;
  border: 1px solid #d9e3f0;
  border-radius: 50%;
  box-shadow: 0 2px 7px rgba(33, 53, 88, 0.06);
  display: flex;
  height: 2.45rem;
  justify-content: center;
  margin-top: 1.9rem;
  overflow: hidden;
  width: 2.45rem;
}
.pb-avatar svg { height: 1.32rem; width: 1.32rem; }
.pb-avatar-user { background: #edf4ff; color: #3f63bf; }
.pb-avatar-assistant { background: #f7faff; color: #3f63bf; }
.pb-avatar-assistant .sq-a { fill: #e66d78; }
.pb-avatar-assistant .sq-b { fill: #5f8fd5; }
.pb-avatar-assistant .sq-c { fill: #6fb6aa; }

.pb-message-header {
  align-items: center;
  display: flex;
  font-size: 0.94rem;
  font-weight: 700;
  gap: 0.9rem;
  line-height: 1.3;
  margin: 0 0.15rem 0.38rem;
}
.pb-user-header { color: #3b5d9b; justify-content: flex-end; }
.pb-assistant-header { color: #31547c; justify-content: flex-start; }
.pb-message-time {
  color: #94a1b2;
  font-size: 0.74rem;
  font-weight: 450;
  margin-left: 0;
}
[class*="st-key-kb-user-bubble-"],
[class*="st-key-paper-user-bubble-"] {
  background: var(--pb-user-bubble);
  border: 1px solid #cfdef5;
  border-radius: 14px 14px 4px 14px;
  box-shadow: 0 2px 5px rgba(47, 82, 139, 0.035);
  margin-left: auto !important;
  max-width: 100% !important;
  padding: 0.82rem 1.02rem;
  width: fit-content !important;
}

[class*="st-key-kb-assistant-bubble-"],
[class*="st-key-paper-assistant-bubble-"] {
  background: #fff;
  border: 1px solid #dde5ef;
  border-radius: 14px 14px 14px 4px;
  box-shadow: 0 4px 14px rgba(35, 55, 88, 0.04);
  padding: 1rem 1.12rem;
}

[class*="st-key-kb-user-bubble-"] [data-testid="stMarkdownContainer"],
[class*="st-key-kb-assistant-bubble-"] [data-testid="stMarkdownContainer"],
[class*="st-key-paper-user-bubble-"] [data-testid="stMarkdownContainer"],
[class*="st-key-paper-assistant-bubble-"] [data-testid="stMarkdownContainer"] {
  color: #26364e;
  font-size: 0.98rem;
  font-weight: 400;
  line-height: 1.78;
}

[class*="st-key-kb-user-bubble-"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-kb-assistant-bubble-"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-paper-user-bubble-"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-paper-assistant-bubble-"] [data-testid="stMarkdownContainer"] p {
  margin-bottom: 0.55rem;
}

.pb-answer-section { margin: 0.06rem 0 0.18rem; }
.pb-answer-section-title {
  color: #315f9c;
  font-size: 1.02rem;
  font-weight: 740;
  line-height: 1.4;
  margin: 0.86rem 0 0.28rem;
}
.pb-answer-section:first-child .pb-answer-section-title { margin-top: 0; }
.pb-answer-section-muted .pb-answer-section-title { color: #546b84; }

/* ---------- Evidence ---------- */
[class*="st-key-kb-assistant-bubble-"] div[data-testid="stExpander"],
[class*="st-key-paper-assistant-bubble-"] div[data-testid="stExpander"] {
  background: #f8faff;
  border-color: #e1e8f1;
  border-radius: 9px;
  margin-top: 0.55rem;
}
.pb-evidence-id { color: #365b88; font-size: 0.8rem; font-weight: 720; margin-bottom: 0.16rem; }
.pb-evidence-meta { color: #8090a4; font-size: 0.74rem; line-height: 1.45; margin-bottom: 0.4rem; }
.pb-evidence-text-label { color: #61748b; font-size: 0.74rem; font-weight: 680; margin-bottom: 0.16rem; }
.pb-evidence-gap { height: 0.28rem; }

/* ---------- Composer ---------- */
div[data-testid="stForm"] {
  background: #f8faff;
  border: 1px solid #e2e8f1;
  border-radius: 12px;
  box-shadow: none;
  margin-top: 0.5rem;
  padding: 0.36rem 0.42rem;
}
div[data-testid="stForm"] input {
  background: #fff !important;
  border: 1px solid #d9e2ee !important;
  box-shadow: none !important;
  color: #27374e;
  font-size: 0.96rem;
  min-height: 2.7rem;
  padding: 0.55rem 0.72rem;
}
div[data-testid="stFormSubmitButton"] button {
  background: #3f63bf !important;
  border-color: #3f63bf !important;
  border-radius: 10px !important;
  box-shadow: 0 3px 9px rgba(63, 99, 191, 0.16) !important;
  color: #fff !important;
  font-size: 1.15rem;
  min-height: 2.7rem;
  padding: 0;
}
div[data-testid="stFormSubmitButton"] button:hover {
  background: #3458b5 !important;
  border-color: #3458b5 !important;
}

/* ---------- Overview ---------- */
.pb-overview-heading { color: #253750; font-size: 1.02rem; font-weight: 720; margin: 0.2rem 0 0.55rem; }
[class*="st-key-paper-overview-scroll-"] {
  height: var(--pb-paper-tab-height) !important;
  max-height: var(--pb-paper-tab-height) !important;
  overflow-y: auto !important;
  padding: 0.55rem 0.18rem 0.8rem;
  scrollbar-width: thin;
  scrollbar-color: #cbd6e5 transparent;
}
[class*="st-key-paper-overview-scroll-"] > [data-testid="stVerticalBlockBorderWrapper"] {
  height: 100% !important;
  max-height: 100% !important;
  overflow-y: auto !important;
  border: 0 !important;
}
[class*="st-key-paper-overview-card-"] {
  background: #fff;
  border: 1px solid #e1e8f1 !important;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(35, 55, 88, 0.028);
  margin-bottom: 0.62rem;
  padding: 0.82rem 0.95rem 0.7rem;
}
.pb-overview-card-title { color: #243b5c; font-size: 1.04rem; font-weight: 730; }
.pb-overview-card-head { align-items: center; display: flex; gap: 0.8rem; justify-content: space-between; margin-bottom: 0.34rem; }
.pb-overview-source-chip { background: #f7f9fc; border: 1px solid #e1e8f1; border-radius: 7px; color: #78889c; font-size: 0.68rem; max-width: 11rem; overflow: hidden; padding: 0.22rem 0.42rem; text-overflow: ellipsis; white-space: nowrap; }
[class*="st-key-paper-overview-card-"] [data-testid="stMarkdownContainer"] { color: #2d3b52; font-size: 0.95rem; line-height: 1.74; }

/* Overview / Explain 的来源卡片 */
[class*="st-key-paper-overview-source-"],
[class*="st-key-explain-source-"] {
  background: #f8faff;
  border-color: #e2e8f1 !important;
  border-radius: 8px;
  margin: 0.4rem 0;
  padding: 0.5rem 0.62rem 0.3rem;
}
.pb-overview-source-meta, .pb-source-meta { color: #7b899c; font-size: 0.74rem; line-height: 1.4; margin-bottom: 0.18rem; }
.pb-overview-source-label, .pb-source-label { color: #5f7187; font-size: 0.75rem; font-weight: 680; margin-bottom: 0.1rem; }

[class*="st-key-paper-overview-source-"] [data-testid="stMarkdownContainer"],
[class*="st-key-explain-source-"] [data-testid="stMarkdownContainer"],
[class*="st-key-kb-assistant-bubble-"] div[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[class*="st-key-paper-assistant-bubble-"] div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
  color: #40516a;
  font-size: 0.88rem;
  line-height: 1.68;
}

[class*="st-key-paper-overview-source-"] .stButton button,
[class*="st-key-explain-source-"] .stButton button {
  min-height: 2.05rem;
  padding: 0.28rem 0.62rem;
}

/* ---------- Explain ---------- */
.st-key-explain-tree-scroll,
.st-key-explain-reader-scroll {
  background: #fbfcfe;
  border: 1px solid #e3e9f1;
  border-radius: 10px;
  height: var(--pb-explain-height) !important;
  max-height: var(--pb-explain-height) !important;
  overflow-y: auto !important;
  padding: 0.62rem 0.68rem;
}
.st-key-explain-tree-scroll > [data-testid="stVerticalBlockBorderWrapper"],
.st-key-explain-reader-scroll > [data-testid="stVerticalBlockBorderWrapper"] {
  height: 100% !important;
  max-height: 100% !important;
  overflow-y: auto !important;
  border: 0 !important;
}
.pb-section-tree-title { color: #53647a; font-size: 0.9rem; font-weight: 720; margin: 0.18rem 0 0.45rem; }
.st-key-paper-section-tree .stButton button {
  background: transparent;
  border-color: transparent;
  color: #52637a;
  justify-content: flex-start;
  min-height: 2.05rem;
  padding: 0.32rem 0.5rem;
  text-align: left;
  white-space: normal;
}
.st-key-paper-section-tree .stButton button:hover { background: #eef4fb; color: #315b98; }
.st-key-paper-section-tree .stButton button[kind="primary"] {
  background: #e9f1ff;
  border-color: #d4e1f7;
  box-shadow: inset 3px 0 0 var(--pb-blue);
  color: #31599a;
}
.pb-explain-title { color: #22344d; font-size: 1.26rem; font-weight: 740; line-height: 1.35; }
.pb-explain-mode { color: #53709a; font-size: 0.78rem; font-weight: 650; margin: 0.22rem 0 0.8rem; }
.pb-explain-subheading { color: #315f9c; font-size: 0.98rem; font-weight: 700; margin-top: 0.95rem; padding-top: 0.5rem; }
.pb-explain-point-title { color: #315f9c; font-size: 0.93rem; font-weight: 720; margin: 0.72rem 0 0.16rem; }
/* 只作用于右侧 Explain 的“解释该章节 / 生成章节概览”主操作按钮。 */
[class*="st-key-paper-explain-generate-"] .stButton button,
[class*="st-key-paper-explain-generate-"] .stButton button * {
  color: #fff !important;
}
.st-key-explain-reader-scroll [data-testid="stMarkdownContainer"] { color: #2d3b52; font-size: 0.96rem; line-height: 1.76; }
.pb-explain-empty { color: #78889b; padding: 14vh 1rem; text-align: center; }

/* ---------- Ask This Paper ---------- */
.pb-ask-title { color: #24344d; font-size: 1.08rem; font-weight: 720; }
.pb-ask-subtitle, .pb-ask-paper { color: #7c8a9d; font-size: 0.78rem; margin-top: 0.14rem; }
.pb-ask-divider { border-top: 1px solid var(--pb-border); margin: 0.65rem 0 0.45rem; }

/* ---------- 小屏：保持可用，而不是把三栏挤到不可读 ---------- */
@media (max-width: 1100px) {
  :root {
    --pb-chat-scroll-height: calc(100dvh - 225px);
  }
  .pb-brand-name { font-size: 1.34rem; }
  .st-key-pb-global-nav { padding-left: 0.72rem; padding-right: 0.72rem; }
  .st-key-pb-context-panel { padding-left: 0.8rem; padding-right: 0.8rem; }
  .st-key-pb-main-panel { padding-left: 0.45rem; padding-right: 0.45rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )
