"""最小非流式 LLM 连通性检查：只验证是否获得最终回答，不输出回答或任何凭证。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from paperbase.config import default_config_path, load_settings

from .client import (
    LLMConfigurationError,
    create_chat_completion_client,
    load_llm_runtime_settings,
)


def main(argv: Sequence[str] | None = None) -> int:
    """按当前 provider 的正式 Client 发送一条固定非流式健康检查问题。"""
    parser = argparse.ArgumentParser(
        description="Run a minimal non-streaming LLM provider health check."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="config.yaml 路径；同目录 .env 中的 LLM 配置会被读取。",
    )
    args = parser.parse_args(argv)

    try:
        app_settings = load_settings(args.config)
        runtime = load_llm_runtime_settings(app_settings.config_path.parent / ".env")
        # 不输出模型回答本身，防止健康检查日志混入未来真实用户内容。
        final_content = create_chat_completion_client(runtime).complete(
            system_prompt="你是健康检查助手。只需简短回复。",
            user_prompt="你好",
        )
    except LLMConfigurationError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2
    except Exception as error:
        # 只报告异常类型，避免 SDK 或代理错误文本中意外包含请求细节。
        print(
            json.dumps(
                {"status": "failed", "error_category": type(error).__name__},
                ensure_ascii=False,
            )
        )
        return 1

    healthy = isinstance(final_content, str) and bool(final_content.strip())
    print(
        json.dumps(
            {
                # 状态：是否成功获得非流式最终回答。
                "status": "ok" if healthy else "failed",
                # 调用模式：固定非流式，用于核对与正式 LLM 客户端一致。
                "mode": "non_streaming",
                "provider": runtime.provider,
                # 模型名称：仅用于确认当前 .env 选择了哪个模型。
                "model": runtime.model,
                # 最终内容状态：是否存在非空 message.content。
                "final_content_received": healthy,
                # 该字段来自实际请求配置，绝不读取或输出隐藏推理文本。
                "thinking_enabled": runtime.thinking_enabled,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
