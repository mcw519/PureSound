#!/bin/bash
set -euo pipefail

uv sync --locked --group dev
uv build
