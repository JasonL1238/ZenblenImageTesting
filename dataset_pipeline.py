#!/usr/bin/env python3
"""Entry point for the smoothie dataset pipeline.

    python dataset_pipeline.py <command> [--dataset smoothie_dataset] [options]

See ``dataset_tools/cli.py`` for the command list, or run with ``-h``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_tools.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
