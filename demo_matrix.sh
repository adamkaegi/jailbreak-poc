#!/usr/bin/env bash
set -euo pipefail

# Demo matrix for Git Bash on Windows.
# Override these when needed, for example:
#   N=10 MODELS="dolphin-mistral:7b,qwen2.5:3b" ./demo_matrix.sh

N="${N:-100}"
MODELS="${MODELS:-dolphin-mistral:7b}"
GUARD_MODEL="${GUARD_MODEL:-llama-guard3:1b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
ATTACKS="${ATTACKS:-none template}"
DEFENSES="${DEFENSES:-none self_reminder llama_guard both}"

run_condition() {
  local attack="$1"
  local defense="$2"

  echo
  echo "============================================================"
  echo "attack=${attack} defense=${defense} n=${N}"
  echo "models=${MODELS}"
  echo "============================================================"

  python run.py \
    --models "${MODELS}" \
    --attack "${attack}" \
    --defense "${defense}" \
    --guard-model "${GUARD_MODEL}" \
    --ollama-url "${OLLAMA_URL}" \
    --n "${N}"
}

echo "Checking local Ollama models..."
if command -v ollama >/dev/null 2>&1; then
  ollama list
else
  powershell.exe -NoProfile -Command "ollama list" || true
fi

for attack in ${ATTACKS}; do
  for defense in ${DEFENSES}; do
    run_condition "${attack}" "${defense}"
  done
done

echo
echo "Latest result files:"
ls -t results/run_*.md results/run_*.csv results/run_*.json 2>/dev/null | head -n 12 || true

echo
echo "Tip: run 'ollama ps' or 'nvidia-smi' in another terminal while this script is running to verify GPU use."
