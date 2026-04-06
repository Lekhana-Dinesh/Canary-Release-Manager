"""
Import-path compatibility helpers.

The project needs to support two import modes:
  1. `server.app:app` from the repository root for OpenEnv/Space-style loading
  2. `canary_release_env.server.app:app` after package installation

This module keeps both modes working without changing the repo layout.
"""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outer_root = repo_root.parent
    for path in (repo_root, outer_root):
        candidate = str(path)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
