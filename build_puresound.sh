#!/bin/bash
set -euo pipefail

uv sync --group dev
uv build
