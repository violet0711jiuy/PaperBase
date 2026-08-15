"""Paper Overview 的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from paperbase.config import default_config_path, load_settings

from .service import run_paper_overview_stage


def main() -> None:
    """为一个已创建的 temporary workspace 生成结构化论文速览。"""
    parser = argparse.ArgumentParser(description="Generate a PaperBase temporary-paper overview.")
    parser.add_argument("workspace_id", help="The staging_<uuid> directory name.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()
    outcome = run_paper_overview_stage(
        settings=load_settings(args.config), workspace_id=args.workspace_id
    )
    print(
        f"overview={outcome.overview_path}\ncontext={outcome.context_path}\n"
        f"selected_chunks={outcome.selected_chunk_count}\n"
        f"paper_title={outcome.overview.paper_title}"
    )


if __name__ == "__main__":
    main()
