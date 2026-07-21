#!/usr/bin/env bash
set -euo pipefail

# Run the mapping cost-comparison workflow safely inside tmux so it survives SSH disconnects.

SESSION_NAME="${SESSION_NAME:-slam_model_consistent_sweep}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NOTEBOOK_REL="notebooks/mapping/cost_comparisons/model_consistent_sweep.ipynb"
PY_SCRIPT_REL="scripts/model_consistent_sweep.py"
RUN_ROOT="${REPO_ROOT}/output/notebook_runs/model_consistent_sweep"
ACTION="${1:-start-script}"
CONVERTER_REL="scripts/convert_notebook_to_script.py"
VENV_PATH="${VENV_PATH:-/global/home/hpc5656/venv_pomdp}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_model_consistent_sweep_tmux.sh [start|start-script|export-script|attach|status|logs|logs-raw|pane|stop]

Actions:
  start         Start notebook execution in detached tmux session
  start-script  Run standalone Python script in detached tmux session
  export-script Convert notebook to .py (one-time/manual refresh)
  attach  Attach to the tmux session
  status  Show tmux session status and latest output files
  logs    Follow latest log (or show live tmux pane if no full log)
  logs-raw Follow latest log without post-processing
  pane    Follow live tmux pane output directly
  stop    Stop the tmux session

Optional env vars:
  SESSION_NAME=custom_name
  NB_EXECUTOR=auto|papermill|nbconvert
  LOG_MODE=pane|full  # for start-script (default: pane)
  VENV_PATH=/global/home/hpc5656/venv_pomdp
  PYTHON_BIN=/global/home/hpc5656/venv_pomdp/bin/python
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

latest_log_target() {
  if [[ -L "${RUN_ROOT}/latest.log" || -e "${RUN_ROOT}/latest.log" ]]; then
    readlink -f "${RUN_ROOT}/latest.log" || true
  fi
}

latest_script_target() {
  if [[ -L "${RUN_ROOT}/latest.script.py" || -e "${RUN_ROOT}/latest.script.py" ]]; then
    readlink -f "${RUN_ROOT}/latest.script.py" || true
  fi
}

prepare_run_dir() {
  mkdir -p "${RUN_ROOT}"
  local ts run_dir log_file
  ts="$(date +%Y%m%d_%H%M%S)"
  run_dir="${RUN_ROOT}/${ts}"
  mkdir -p "${run_dir}"
  log_file="${run_dir}/run.log"
  ln -sfn "${run_dir}" "${RUN_ROOT}/latest"
  ln -sfn "${log_file}" "${RUN_ROOT}/latest.log"
  printf "%s\n%s\n" "${run_dir}" "${log_file}"
}

start_run() {
  require_cmd tmux
  require_cmd python

  if session_exists; then
    echo "Session '${SESSION_NAME}' is already running."
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
  fi

  local run_dir log_file
  mapfile -t run_info < <(prepare_run_dir)
  run_dir="${run_info[0]}"
  log_file="${run_info[1]}"
  local ts
  ts="$(basename "${run_dir}")"

  local output_nb="model_consistent_sweep.executed.${ts}.ipynb"
  local output_path="${run_dir}/${output_nb}"

  ln -sfn "${output_path}" "${RUN_ROOT}/latest.executed.ipynb"

  local requested_executor="${NB_EXECUTOR:-auto}"
  local nb_exec=""
  local executor_used=""
  if [[ "${requested_executor}" == "auto" || "${requested_executor}" == "papermill" ]]; then
    if command -v papermill >/dev/null 2>&1; then
      # Papermill streams cell outputs to terminal and writes output notebook progressively.
      nb_exec="papermill \"${NOTEBOOK_REL}\" \"${output_path}\" \
--progress-bar \
--log-output \
--request-save-on-cell-execute \
--autosave-cell-every 30"
      executor_used="papermill"
    elif [[ "${requested_executor}" == "papermill" ]]; then
      echo "Error: NB_EXECUTOR=papermill requested but 'papermill' not found in PATH." >&2
      echo "Install with: python -m pip install papermill" >&2
      exit 1
    fi
  fi

  if [[ -z "${nb_exec}" ]] && [[ "${requested_executor}" == "auto" || "${requested_executor}" == "nbconvert" ]]; then
    if command -v jupyter >/dev/null 2>&1; then
      nb_exec="jupyter nbconvert --to notebook --execute \
--ExecutePreprocessor.timeout=-1 \
--output \"${output_nb}\" --output-dir \"${run_dir}\" \
\"${NOTEBOOK_REL}\""
      executor_used="nbconvert(jupyter)"
    elif python -c "import nbconvert" >/dev/null 2>&1; then
      nb_exec="python -m nbconvert --to notebook --execute \
--ExecutePreprocessor.timeout=-1 \
--output \"${output_nb}\" --output-dir \"${run_dir}\" \
\"${NOTEBOOK_REL}\""
      executor_used="nbconvert(python -m)"
    elif [[ "${requested_executor}" == "nbconvert" ]]; then
      echo "Error: NB_EXECUTOR=nbconvert requested but no nbconvert executor found." >&2
      echo "Install with: python -m pip install jupyter nbconvert" >&2
      exit 1
    fi
  fi

  if [[ -z "${nb_exec}" ]]; then
    echo "Error: no notebook executor found in current environment." >&2
    echo "Recommended install (for better live output): python -m pip install papermill" >&2
    echo "Fallback install: python -m pip install jupyter nbconvert" >&2
    exit 1
  fi

  local cmd
  if command -v script >/dev/null 2>&1; then
    # Keep a PTY for notebook execution so tqdm-style progress bars update naturally.
    cmd="cd \"${REPO_ROOT}\" \
&& export PYDEVD_DISABLE_FILE_VALIDATION=1 \
&& script -q -f \"${log_file}\" -c '${nb_exec}'"
  else
    # Fallback when util-linux script is unavailable.
    cmd="cd \"${REPO_ROOT}\" \
&& export PYDEVD_DISABLE_FILE_VALIDATION=1 \
&& ${nb_exec} \
2>&1 | tee \"${log_file}\""
  fi

  tmux new-session -d -s "${SESSION_NAME}" "${cmd}"

  echo "Started session: ${SESSION_NAME}"
  echo "Notebook: ${NOTEBOOK_REL}"
  echo "Output notebook: ${output_path}"
  echo "Log file: ${log_file}"
  echo "Executor: ${executor_used}"
  echo
  echo "Useful commands:"
  echo "  tmux attach -t ${SESSION_NAME}"
  echo "  scripts/run_model_consistent_sweep_tmux.sh logs"
  echo "  scripts/run_model_consistent_sweep_tmux.sh status"
}

export_script_run() {
  require_cmd python
  local output_rel="${2:-scripts/generated/model_consistent_sweep.py}"
  local output_abs="${REPO_ROOT}/${output_rel}"
  cd "${REPO_ROOT}"
  python "${CONVERTER_REL}" "${NOTEBOOK_REL}" "${output_abs}"
  echo "Exported script: ${output_abs}"
}

start_script_run() {
  require_cmd tmux
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN or VENV_PATH to your desired environment." >&2
    exit 1
  fi

  if session_exists; then
    echo "Session '${SESSION_NAME}' is already running."
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
  fi

  local run_dir log_file
  mapfile -t run_info < <(prepare_run_dir)
  run_dir="${run_info[0]}"
  log_file="${run_info[1]}"
  local script_out="${REPO_ROOT}/${PY_SCRIPT_REL}"
  local mode="${LOG_MODE:-pane}"
  ln -sfn "${script_out}" "${RUN_ROOT}/latest.script.py"

  local start_script="${run_dir}/start_script.sh"
  cat > "${start_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
if [[ ! -f "${script_out}" ]]; then
  echo "Script not found: ${script_out}" >&2
  echo "Generate it with: scripts/run_model_consistent_sweep_tmux.sh export-script" >&2
  exit 1
fi
echo "[\$(date -Is)] run-script=${script_out}" >> "${log_file}"
echo "[\$(date -Is)] python-bin=${PYTHON_BIN}" >> "${log_file}"
if "${PYTHON_BIN}" -c "import cupy as cp; print(cp.__version__)" >> "${log_file}" 2>&1; then
  echo "[\$(date -Is)] cupy-check=ok" >> "${log_file}"
else
  echo "[\$(date -Is)] cupy-check=failed" >> "${log_file}"
fi
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="\${TQDM_MININTERVAL:-1.0}"
if [[ "${mode}" == "full" ]]; then
  if command -v script >/dev/null 2>&1; then
    exec script -q -f "${log_file}" -c "\"${PYTHON_BIN}\" -u \"${script_out}\""
  else
    exec "${PYTHON_BIN}" -u "${script_out}" 2>&1 | tee -a "${log_file}"
  fi
else
  echo "[\$(date -Is)] running-in-pane-mode" >> "${log_file}"
  # Pane mode keeps tqdm redraw behavior terminal-native and avoids huge log growth.
  # We still mirror stderr to run.log for post-mortem tracebacks.
  set +e
  "${PYTHON_BIN}" -u "${script_out}" 2> >(tee -a "${log_file}" >&2)
  rc=\$?
  echo "[\$(date -Is)] process-exit-code=\${rc}" >> "${log_file}"
  if [[ \${rc} -ne 0 ]]; then
    echo "[\$(date -Is)] process-failed" >> "${log_file}"
  else
    echo "[\$(date -Is)] process-succeeded" >> "${log_file}"
  fi
  exit \${rc}
fi
EOF
  chmod +x "${start_script}"

  tmux new-session -d -s "${SESSION_NAME}" "bash \"${start_script}\""

  echo "Started session: ${SESSION_NAME}"
  echo "Script: ${script_out}"
  echo "Mode: ${mode}"
  echo "Metadata log: ${log_file}"
  echo
  echo "Useful commands:"
  echo "  tmux attach -t ${SESSION_NAME}"
  echo "  scripts/run_model_consistent_sweep_tmux.sh pane"
  echo "  scripts/run_model_consistent_sweep_tmux.sh status"
  echo "  rg \"process-exit-code|process-failed|process-succeeded\" \"${log_file}\""
}

attach_run() {
  require_cmd tmux
  if ! session_exists; then
    echo "No running session '${SESSION_NAME}'."
    exit 1
  fi
  tmux attach -t "${SESSION_NAME}"
}

status_run() {
  require_cmd tmux
  if session_exists; then
    echo "Session '${SESSION_NAME}' is running."
    tmux list-sessions | rg "^${SESSION_NAME}:"
  else
    echo "Session '${SESSION_NAME}' is not running."
  fi

  if [[ -L "${RUN_ROOT}/latest" ]]; then
    echo "Latest run dir: $(readlink -f "${RUN_ROOT}/latest")"
  fi
  if [[ -L "${RUN_ROOT}/latest.log" || -e "${RUN_ROOT}/latest.log" ]]; then
    echo "Latest log: $(latest_log_target)"
  fi
  if [[ -L "${RUN_ROOT}/latest.executed.ipynb" ]]; then
    echo "Latest executed notebook: $(readlink -f "${RUN_ROOT}/latest.executed.ipynb")"
  fi
  if [[ -L "${RUN_ROOT}/latest.script.py" || -e "${RUN_ROOT}/latest.script.py" ]]; then
    echo "Latest exported script: $(latest_script_target)"
  fi
}

logs_run() {
  local log_target
  log_target="$(latest_log_target || true)"
  if [[ -z "${log_target}" || ! -e "${log_target}" ]]; then
    echo "No latest log found."
    echo "Use pane mode for live progress:"
    echo "  scripts/run_model_consistent_sweep_tmux.sh pane"
    exit 1
  fi
  tail -n 80 "${log_target}"
  if session_exists; then
    echo "---- live follow (CR->NL) ----"
    python -u - "${log_target}" <<'PY'
import os
import sys
import time

path = sys.argv[1]
with open(path, "rb") as f:
    f.seek(0, os.SEEK_END)
    while True:
        chunk = f.read(4096)
        if chunk:
            text = chunk.decode("utf-8", errors="replace").replace("\r", "\n")
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            time.sleep(0.2)
PY
  fi
}

logs_raw_run() {
  local log_target
  log_target="$(latest_log_target || true)"
  if [[ -z "${log_target}" || ! -e "${log_target}" ]]; then
    echo "No latest log found."
    exit 1
  fi
  tail -f "${log_target}"
}

pane_run() {
  require_cmd tmux
  if ! session_exists; then
    echo "No running session '${SESSION_NAME}'."
    exit 1
  fi
  while true; do
    clear
    tmux capture-pane -pt "${SESSION_NAME}":0 -S -200
    sleep 2
  done
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
  start-script) start_script_run ;;
  export-script) export_script_run "$@" ;;
  attach) attach_run ;;
  status) status_run ;;
  logs) logs_run ;;
  logs-raw) logs_raw_run ;;
  pane) pane_run ;;
  stop) stop_run ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    echo
    usage
    exit 1
    ;;
esac

