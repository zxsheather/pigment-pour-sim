#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Use the documented Python 3.11 environment.')
PY
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r deploy/requirements.txt
python -m deploy.artifacts source
python -m pip check
echo 'Runtime ready. Next: python -m deploy.artifacts download'
