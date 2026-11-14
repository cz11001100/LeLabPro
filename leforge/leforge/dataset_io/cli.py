# Copyright 2026 The LeForge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Command-line entry point for testing conversions without the UI.

Examples:
    python -m leforge.dataset_io.cli inspect --path /data/aloha
    python -m leforge.dataset_io.cli import --preset aloha_hdf5 \
        --source /data/aloha --repo-id me/demo
    python -m leforge.dataset_io.cli export --preset zarr_diffusion_policy \
        --repo-id me/demo --out /data/demo.zarr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .jobs import ExportRequest, ImportRequest, run_export_sync, run_import_sync
from .formats import inspect_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LeForge dataset format converter")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Detect and summarize a source dataset")
    inspect_p.add_argument("--path", required=True)
    inspect_p.add_argument("--preset", default=None)

    import_p = sub.add_parser("import", help="Convert an external format into LeRobotDataset v3")
    import_p.add_argument("--preset", required=True)
    import_p.add_argument("--source", required=True)
    import_p.add_argument("--repo-id", required=True)
    import_p.add_argument("--fps", type=float, default=None)
    import_p.add_argument("--task", default=None)
    import_p.add_argument("--state-key", default=None)
    import_p.add_argument("--action-key", default=None)
    import_p.add_argument("--no-video", action="store_true")

    export_p = sub.add_parser("export", help="Export a LeRobotDataset to an external format")
    export_p.add_argument("--preset", required=True)
    export_p.add_argument("--repo-id", required=True)
    export_p.add_argument("--out", required=True)
    export_p.add_argument("--quality", type=int, default=95)

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_source(args.path, args.preset)
        elif args.command == "import":
            result = run_import_sync(
                ImportRequest(
                    preset=args.preset,
                    source_path=args.source,
                    target_repo_id=args.repo_id,
                    fps=args.fps,
                    task=args.task,
                    state_key=args.state_key,
                    action_key=args.action_key,
                    video=not args.no_video,
                )
            )
        else:
            result = run_export_sync(
                ExportRequest(
                    preset=args.preset,
                    repo_id=args.repo_id,
                    output_path=args.out,
                    quality=args.quality,
                )
            )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
