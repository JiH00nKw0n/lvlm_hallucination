#!/usr/bin/env bash
# Container entrypoint. Reads CONFIG and STAGE from env, dispatches to run.py.
set -euo pipefail

CONFIG=${CONFIG:?"set CONFIG=clean/configs/<...>.yaml"}
STAGE=${STAGE:-all}

cd /app
echo "[entrypoint] CONFIG=$CONFIG STAGE=$STAGE"
exec python -m clean.run "$CONFIG" --stage "$STAGE"
