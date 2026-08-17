"""Explain Section 的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from paperbase.config import default_config_path, load_settings

from .service import run_explain_section_stage


def main() -> None:
    """为一个 temporary workspace 的指定 section 生成一次 Explain JSON。"""
    parser = argparse.ArgumentParser(description="Explain a section in a PaperBase temporary workspace.")
    parser.add_argument("workspace_id", help="The staging_<uuid> directory name.")
    parser.add_argument("section_id", help="A section_id stored in parsed/parsed_paper.json.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()
    outcome = run_explain_section_stage(
        settings=load_settings(args.config),
        workspace_id=args.workspace_id,
        section_id=args.section_id,
    )
    print(
        f"explain={outcome.explanation_path}\ncontext={outcome.context_path}\n"
        f"mode={outcome.explanation.mode}\nselected_chunks={outcome.selected_chunk_count}\n"
        f"tokens={outcome.context.final_token_count}/{outcome.context.max_token_budget}"
    )


if __name__ == "__main__":
    main()
