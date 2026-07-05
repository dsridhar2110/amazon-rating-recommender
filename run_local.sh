#!/usr/bin/env bash
# RecoPulse — start the full stack locally (model API + web UI).
# Usage:  ./run_local.sh      then open http://localhost:3000
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$ROOT/ml/artifacts/model.joblib" ]; then
  echo "No trained model found. Training first (a few minutes)..."
  "$ROOT/venv/bin/python" "$ROOT/ml/train.py"
fi

echo "Starting model API on :8000 ..."
"$ROOT/venv/bin/uvicorn" ml.api:app --port 8000 --log-level warning &
API_PID=$!

echo "Starting web UI on :3000 ..."
( cd "$ROOT/web" && npm run dev -- --port 3000 ) &
WEB_PID=$!

trap "echo; echo 'Stopping...'; kill $API_PID $WEB_PID 2>/dev/null" INT TERM
echo
echo "  ▶  RecoPulse is up:  http://localhost:3000   (API: http://localhost:8000/health)"
echo "     Ctrl-C to stop both."
wait
