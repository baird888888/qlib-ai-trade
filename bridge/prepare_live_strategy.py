from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bridge.live_control import prepare_strategy_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare live signal files from a chosen search summary.")
    parser.add_argument(
        "--summary",
        default=str((Path(__file__).resolve().parents[1] / "bridge" / "runtime" / "reports" / "live_strategy_search_summary.json")),
        help="Path to a *search_summary.json file.",
    )
    args = parser.parse_args()

    payload = prepare_strategy_signals(args.summary)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
