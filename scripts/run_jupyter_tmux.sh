#!/usr/bin/env bash
set -euo pipefail

# Start/monitor a persistent Jupyter Lab server in tmux.

SESSION_NAME="${SESSION_NAME:-slam_jupyter_lab}"
PORT="${PORT:-8888}"
IP="${IP:-127.0.0.1}"
VENV_PATH="${VENV_PATH:-/global/home/hpc5656/venv_pomdp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${REPO_ROOT}/output/jupyter_server"
ACTION="${1:-start}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_jupyter_tmux.sh [start|status|attach|logs|url|stop]

Actions:
  start   Start Jupyter Lab in detached tmux session
  status  Show tmux session status and latest log path
  attach  Attach to tmux session
  logs    Follow Jupyter server log
  url     Print detected local Jupyter URL from log
  stop    Stop tmux session

Optional env vars:
  SESSION_NAME=custom_session
  PORT=8888
  IP=127.0.0.1
  VENV_PATH=/global/home/hpc5656/venv_pomdp
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH." >&2
    exit 1
  fi
}

session_exists() {
  tmux has-session -t "${SESSION_NAME}" >/dev/null 2>&1
}

latest_log_path() {
  local p="${RUN_ROOT}/latest.log"
  if [[ -L "${p}" || -e "${p}" ]]; then
    readlink -f "${p}" || true
  fi
}

print_url_from_log() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
text = path.read_text(errors="replace")
matches = re.findall(r"https?://127\.0\.0\.1:\d+/(?:lab|tree)\?token=[A-Za-z0-9]+", text)
if matches:
    print(matches[-1])
PY
}

start_run() {
  require_cmd tmux
  require_cmd bash

  if session_exists; then
    echo "Session '${SESSION_NAME}' is already running."
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
  fi

  mkdir -p "${RUN_ROOT}"
  local ts run_dir log_file
  ts="$(date +%Y%m%d_%H%M%S)"
  run_dir="${RUN_ROOT}/${ts}"
  mkdir -p "${run_dir}"
  log_file="${run_dir}/jupyter.log"

  ln -sfn "${run_dir}" "${RUN_ROOT}/latest"
  ln -sfn "${log_file}" "${RUN_ROOT}/latest.log"

  local jupyter_bin start_script
  if [[ -x "${VENV_PATH}/bin/jupyter" ]]; then
    jupyter_bin="${VENV_PATH}/bin/jupyter"
  else
    require_cmd jupyter
    jupyter_bin="$(command -v jupyter)"
  fi
  start_script="${run_dir}/start_jupyter.sh"
  cat > "${start_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
"${jupyter_bin}" lab --no-browser --ip="${IP}" --port="${PORT}" --port-retries=50 2>&1 | tee "${log_file}"
EOF
  chmod +x "${start_script}"

  tmux new-session -d -s "${SESSION_NAME}" "bash \"${start_script}\""

  echo "Started session: ${SESSION_NAME}"
  echo "Log file: ${log_file}"
  echo "Server bind: ${IP}:${PORT} (with port-retries=50)"
  echo
  echo "Useful commands:"
  echo "  scripts/run_jupyter_tmux.sh logs"
  echo "  scripts/run_jupyter_tmux.sh url"
  echo "  scripts/run_jupyter_tmux.sh attach"
  echo "  scripts/run_jupyter_tmux.sh status"
}

status_run() {
  require_cmd tmux
  if session_exists; then
    echo "Session '${SESSION_NAME}' is running."
    tmux list-sessions | rg "^${SESSION_NAME}:"
  else
    echo "Session '${SESSION_NAME}' is not running."
  fi
  local lp
  lp="$(latest_log_path || true)"
  if [[ -n "${lp}" ]]; then
    echo "Latest log: ${lp}"
  fi
}

attach_run() {
  require_cmd tmux
  if ! session_exists; then
    echo "No running session '${SESSION_NAME}'."
    exit 1
  fi
  tmux attach -t "${SESSION_NAME}"
}

logs_run() {
  local log_target="${RUN_ROOT}/latest.log"
  if [[ ! -e "${log_target}" ]]; then
    echo "No latest log found at ${log_target}."
    exit 1
  fi
  tail -f "${log_target}"
}

url_run() {
  local log_target
  log_target="$(latest_log_path || true)"
  if [[ -z "${log_target}" || ! -e "${log_target}" ]]; then
    echo "No log file found."
    exit 1
  fi

  local detected
  detected="$(print_url_from_log "${log_target}" || true)"
  if [[ -n "${detected}" ]]; then
    echo "${detected}"
    echo "Tunnel command:"
    echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} <user>@<host>"
  else
    echo "No URL detected yet. Jupyter may still be starting."
    exit 1
  fi
}

stop_run() {
  require_cmd tmux
  if ! session_exists; then
    echo "No running session '${SESSION_NAME}'."
    exit 0
  fi
  tmux kill-session -t "${SESSION_NAME}"
  echo "Stopped session '${SESSION_NAME}'."
}

case "${ACTION}" in
  start) start_run ;;
  status) status_run ;;
  attach) attach_run ;;
  logs) logs_run ;;
  url) url_run ;;
  stop) stop_run ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    echo
    usage
    exit 1
    ;;
esac

