#!/usr/bin/env bash
#SBATCH --job-name=k3-sglang
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH -A ASC26075
#SBATCH -p debug
#SBATCH --mail-user=sli@tacc.utexas.edu  
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --exclude=c108-009,c106-029

root_dir=${KIMI_K3_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}
job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
export KIMI_K3_ROOT="$root_dir"
export KIMI_JOB_LOG_DIR="$root_dir/logs/$job_id"
mkdir -p "$KIMI_JOB_LOG_DIR"
exec >"$KIMI_JOB_LOG_DIR/k3-server.out" 2>&1
cd "$KIMI_JOB_LOG_DIR"

module load gcc
module load cuda
module load tacc-apptainer

set -euo pipefail

export KIMI_MODEL_PATH=${KIMI_MODEL_PATH:-$KIMI_K3_ROOT/models/Kimi-K3}
export KIMI_STARTUP_TIMEOUT=${KIMI_STARTUP_TIMEOUT:-7200}
export KIMI_HELLO_TIMEOUT=${KIMI_HELLO_TIMEOUT:-1800}
export KIMI_GPU_MONITOR_INTERVAL=${KIMI_GPU_MONITOR_INTERVAL:-15}
[[ "$KIMI_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    echo "KIMI_STARTUP_TIMEOUT must be a positive integer." >&2
    exit 2
}
[[ "$KIMI_HELLO_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    echo "KIMI_HELLO_TIMEOUT must be a positive integer." >&2
    exit 2
}
[[ "$KIMI_GPU_MONITOR_INTERVAL" =~ ^[1-9][0-9]*$ ]] || {
    echo "KIMI_GPU_MONITOR_INTERVAL must be a positive integer." >&2
    exit 2
}

bash "$KIMI_K3_ROOT/scripts/preflight.sh"

export MASTER_ADDR
# Updated InfiniBand interface for Horizon
MASTER_ADDR=$(ip -4 -o addr show ibs2 | awk '{sub(/\/.*/, "", $4); print $4; exit}')
[[ -n "$MASTER_ADDR" ]] || { echo "Could not resolve rank-0 ibs2 address." >&2; exit 2; }

echo "K3 rank-0 host: $(hostname)"
echo "K3 rank-0 IB address: $MASTER_ADDR"
echo "OpenAI-compatible endpoint after startup: http://$(hostname):30000/v1"
echo "Startup timeout: ${KIMI_STARTUP_TIMEOUT}s; Hello timeout: ${KIMI_HELLO_TIMEOUT}s"
echo "This four-hour 8-node validation job costs at most 32 Horizon SUs."

gpu_csv="$KIMI_JOB_LOG_DIR/k3-gpu.csv"
gpu_monitor_log="$KIMI_JOB_LOG_DIR/k3-gpu-monitor.out"
bash "$KIMI_K3_ROOT/scripts/monitor_k3_gpu.sh" \
    "$SLURM_JOB_ID" \
    "$SLURM_JOB_NODELIST" \
    "$KIMI_GPU_MONITOR_INTERVAL" \
    >"$gpu_monitor_log" 2>&1 &
gpu_monitor_pid=$!
echo "GPU monitoring: every ${KIMI_GPU_MONITOR_INTERVAL}s -> $gpu_csv"
echo "GPU monitor log: $gpu_monitor_log"

srun \
    --nodes=8 \
    --ntasks=8 \
    --ntasks-per-node=1 \
    --cpus-per-task=72 \
    --kill-on-bad-exit=1 \
    --output="$KIMI_JOB_LOG_DIR/k3-rank-%t.out" \
    bash "$KIMI_K3_ROOT/scripts/launch_k3_rank.sh" &
server_step_pid=$!

cleanup() {
    if kill -0 "$server_step_pid" 2>/dev/null; then
        kill "$server_step_pid" 2>/dev/null || true
        wait "$server_step_pid" 2>/dev/null || true
    fi
    if kill -0 "$gpu_monitor_pid" 2>/dev/null; then
        kill "$gpu_monitor_pid" 2>/dev/null || true
        wait "$gpu_monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ready=0
deadline=$((SECONDS + KIMI_STARTUP_TIMEOUT))
while (( SECONDS < deadline )); do
    if curl --silent --show-error --fail --max-time 60 \
        "http://127.0.0.1:30000/health_generate" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$server_step_pid" 2>/dev/null; then
        echo "K3 server step exited before becoming healthy." >&2
        server_status=0
        wait "$server_step_pid" || server_status=$?
        if [[ "$server_status" -eq 0 ]]; then
            server_status=1
        fi
        exit "$server_status"
    fi
    sleep 10
done

if [[ "$ready" -ne 1 ]]; then
    echo "Timed out after ${KIMI_STARTUP_TIMEOUT}s waiting for K3 warmup readiness." >&2
    exit 1
fi

echo "K3 warmup and generation health check passed; running the Hello inference test."
python3 "$KIMI_K3_ROOT/scripts/client.py" \
    --base-url http://127.0.0.1:30000/v1 \
    --prompt Hello \
    --max-tokens 1 \
    --timeout "$KIMI_HELLO_TIMEOUT"
echo "K3_HELLO_INFERENCE_TEST=PASS"
echo "K3 is ready for user requests and will remain active until job termination."

server_status=0
wait "$server_step_pid" || server_status=$?
if kill -0 "$gpu_monitor_pid" 2>/dev/null; then
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
fi
trap - EXIT
exit "$server_status"