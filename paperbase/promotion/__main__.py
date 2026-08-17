"""Add to Knowledge Base 的命令行入口。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from paperbase.config import default_config_path, load_settings

from .service import promote_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a staging PaperBase workspace into the formal KB.")
    parser.add_argument("workspace_id")
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()
    result = promote_workspace(settings=load_settings(args.config), workspace_id=args.workspace_id)
    payload = asdict(result)
    payload["formal_pdf_path"] = str(result.formal_pdf_path) if result.formal_pdf_path else None
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
