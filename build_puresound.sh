#!/bin/bash
set -euo pipefail

uv sync --locked --group dev --extra cpu
uv build
