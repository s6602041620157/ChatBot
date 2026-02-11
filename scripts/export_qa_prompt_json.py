#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa_prompt_manager import QAPromptManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Export QA.xlsx to qa_prompt.json")
    parser.add_argument("--excel", default="data/QA.xlsx", help="Path to source QA Excel file")
    parser.add_argument("--output", default="data/qa_prompt.json", help="Path to output JSON file")
    parser.add_argument(
        "--preserve-answers",
        action="store_true",
        help="Preserve non-empty answers from existing output JSON",
    )
    args = parser.parse_args()

    manager = QAPromptManager(
        excel_path=args.excel,
        json_path=args.output,
        embedding_model=None,
    )
    result = manager.sync_from_excel(force=True, preserve_answers=args.preserve_answers)

    print("QA export completed")
    print(f"  updated: {result['updated']}")
    print(f"  count: {result['count']}")
    print(f"  excel: {result['excel_path']}")
    print(f"  output: {result['json_path']}")


if __name__ == "__main__":
    main()
