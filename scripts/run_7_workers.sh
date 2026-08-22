#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

NUM_RUNS="${NUM_RUNS:-7}"
NUM_TURNS="${NUM_TURNS:-10}"
NUM_WORKERS=7
MASTER_SEED="${MASTER_SEED:-20260821}"
MODEL="${MODEL:-qwen3:8b}"
OLLAMA_PORT_BASE="${OLLAMA_PORT_BASE:-11434}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_DIR}/artifacts/mvp-${NUM_RUNS}}"

mkdir -p "${EXPERIMENT_DIR}/workers" "${EXPERIMENT_DIR}/logs"

for worker_id in $(seq 0 6); do
    port=$((OLLAMA_PORT_BASE + worker_id))
    tags_url="http://127.0.0.1:${port}/api/tags"

    if ! model_json="$(curl --fail --silent --show-error --max-time 10 "${tags_url}")"; then
        echo "Ollama preflight failed for worker ${worker_id}: ${tags_url}" >&2
        exit 1
    fi

    if ! printf '%s' "${model_json}" | "${PYTHON_BIN}" -c '
import json
import sys

requested = sys.argv[1]
models = json.load(sys.stdin).get("models", [])
names = {model.get("name") for model in models}
if requested not in names:
    raise SystemExit(f"model {requested!r} is not available; found {sorted(names)}")
' "${MODEL}"; then
        echo "Model preflight failed for worker ${worker_id}" >&2
        exit 1
    fi
done

progress_metadata="${EXPERIMENT_DIR}/progress.json"
if [[ ! -e "${progress_metadata}" ]]; then
    progress_metadata_tmp="${progress_metadata}.tmp.$$"
    started_at_epoch="$(date -u +%s)"
    printf '{"started_at_epoch":%s,"num_runs":%s,"num_turns":%s,"num_workers":%s}\n' \
        "${started_at_epoch}" \
        "${NUM_RUNS}" \
        "${NUM_TURNS}" \
        "${NUM_WORKERS}" \
        >"${progress_metadata_tmp}"
    mv "${progress_metadata_tmp}" "${progress_metadata}"
fi

pids=()
worker_ids=()

for worker_id in $(seq 0 6); do
    port=$((OLLAMA_PORT_BASE + worker_id))
    output_dir="${EXPERIMENT_DIR}/workers/worker-${worker_id}/runs"
    log_file="${EXPERIMENT_DIR}/logs/worker-${worker_id}.log"

    echo "Starting worker ${worker_id}: port=${port}, output=${output_dir}"

    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${REPO_DIR}/batch_simulation.py" \
        --ollama-url "http://127.0.0.1:${port}/api/chat" \
        --model "${MODEL}" \
        --num-runs "${NUM_RUNS}" \
        --num-turns "${NUM_TURNS}" \
        --master-seed "${MASTER_SEED}" \
        --worker-id "${worker_id}" \
        --num-workers "${NUM_WORKERS}" \
        --output-dir "${output_dir}" \
        >"${log_file}" 2>&1 &

    pids+=("$!")
    worker_ids+=("${worker_id}")
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
        echo "Worker ${worker_ids[${index}]} completed."
    else
        echo "Worker ${worker_ids[${index}]} failed; see its log." >&2
        failed=1
    fi
done

if ((failed != 0)); then
    echo "At least one worker failed; merge was not run." >&2
    exit 1
fi

"${PYTHON_BIN}" "${REPO_DIR}/merge_results.py" \
    --input-root "${EXPERIMENT_DIR}/workers" \
    --output-dir "${EXPERIMENT_DIR}/merged" \
    --num-runs "${NUM_RUNS}" \
    --num-turns "${NUM_TURNS}" \
    --num-workers "${NUM_WORKERS}"

echo "Seven-worker simulation and merge completed: ${EXPERIMENT_DIR}"
