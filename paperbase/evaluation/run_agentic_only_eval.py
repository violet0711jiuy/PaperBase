"""Evaluate the Agentic RAG service without a separate one-shot Retriever run."""

from __future__ import annotations

from .run_agentic_eval import main as run_agentic_eval


def main(argv=None) -> int:
    return run_agentic_eval(argv, agentic_only=True)


if __name__ == "__main__":
    raise SystemExit(main())
