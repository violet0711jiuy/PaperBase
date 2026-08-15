"""临时论文工作区的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from paperbase.config import default_config_path, load_settings

from .service import run_temporary_workspace_stage


def main() -> None:
    """接收一篇新 PDF，并仅在 storage/staging 下创建临时工作区。"""
    argument_parser = argparse.ArgumentParser(
        description="Create an isolated PaperBase temporary paper workspace."
    )
    argument_parser.add_argument("pdf", type=Path)
    argument_parser.add_argument("--config", type=Path, default=default_config_path())
    args = argument_parser.parse_args()
    workspace = run_temporary_workspace_stage(
        settings=load_settings(args.config), source_pdf=args.pdf
    )
    print(
        f"workspace_id={workspace.workspace_id}, paper_id={workspace.paper_id}, "
        f"chunks={workspace.total_chunk_count}, searchable={workspace.searchable_chunk_count}\n"
        f"workspace={workspace.root_dir}\nindex={workspace.index_path}"
    )


if __name__ == "__main__":
    main()
