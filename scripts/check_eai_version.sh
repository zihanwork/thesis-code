#!/usr/bin/env bash
# 步骤 1：对齐 eai-eval 与官方仓库声明的版本
# 用法：CONDA_ENV=eai-eval ./scripts/check_eai_version.sh
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-eai-eval}"
EXPECTED="${EXPECTED_VERSION:-1.0.5}"

echo "=== Conda env: ${CONDA_ENV} ==="
conda run -n "${CONDA_ENV}" python - <<'PY'
import importlib.metadata as m
try:
    v = m.version("eai-eval")
    print("pip eai-eval version:", v)
except Exception as e:
    print("eai-eval not found via importlib.metadata:", e)
PY

echo ""
echo "=== pip show eai-eval ==="
conda run -n "${CONDA_ENV}" pip show eai-eval | grep -E '^(Name|Version|Location):' || true

echo ""
echo "=== 本地 embodied-agent-interface-main/setup.py 声明版本（对照用）==="
REPO_SETUP="$(dirname "$0")/../embodied-agent-interface-main/setup.py"
if [[ -f "${REPO_SETUP}" ]]; then
  grep -E "^\s*version=" "${REPO_SETUP}" || true
else
  echo "(未找到 ${REPO_SETUP}，可忽略)"
fi

echo ""
echo "期望与仓库一致时可对照: eai-eval == ${EXPECTED}"
echo "若版本不一致：优先 pip install -e embodied-agent-interface-main 或 pip install eai-eval==${EXPECTED}"
