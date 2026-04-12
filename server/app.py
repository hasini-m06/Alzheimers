"""
server/app.py
─────────────
Re-exports the root app for OpenEnv multi-mode deployment compliance.
"""

import sys
import os

# Ensure the repo root is on the path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, main  # noqa: F401

if __name__ == "__main__":
    main()