#!/usr/bin/env bash
# Starts the recorder UI at http://127.0.0.1:8765 (venv only; nothing is installed system-wide).
cd "$(dirname "$0")"
exec .venv/bin/python server.py "$@"
