#!/usr/bin/env bash
# ==============================================================================
# Lanzador de 1 comando para el MVP Integrado Indra Smart Pick & Pack (Reto 4)
# Uso:
#   ./scripts/run_mvp.sh           # Ejecución local en Python (.venv)
#   ./scripts/run_mvp.sh --docker  # Ejecución en contenedores Docker Compose
# ==============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "--docker" ]]; then
  echo "[Docker Mode] Levantando servicios con docker compose..."
  docker compose up --build
  exit 0
fi

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

echo "[1/2] Iniciando Servicio de Inferencia de Visión en http://localhost:8000 ..."
"$PYTHON_BIN" -m uvicorn src.vision_service.app:app --host 0.0.0.0 --port 8000 &
VISION_PID=$!

trap 'kill $VISION_PID 2>/dev/null || true' EXIT
sleep 2

echo "[2/2] Iniciando Copiloto Agéntico + Consola Web en http://localhost:8001 ..."
echo "      -> Consola Operativa Interactiva: http://localhost:8001/"
"$PYTHON_BIN" -m uvicorn src.agent_service.app:app --host 0.0.0.0 --port 8001
