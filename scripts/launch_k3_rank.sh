#!/usr/bin/env bash


set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=${KIMI_K3_ROOT:-$(cd "$script_dir/.." && pwd)}
model_path=${KIMI_MODEL_PATH:-$root_dir/models/Kimi-K3}
master_addr=${MASTER_ADDR:?MASTER_ADDR must be the rank-0 InfiniBand address}
node_rank=${SLURM_PROCID:?This script must run under srun}
node_count=${SLURM_NNODES:?This script must run inside a Slurm job}
model_load_threads=${KIMI_MODEL_LOAD_THREADS:-2}

if [[ "$node_count" -ne 8 ]]; then
    echo "The production K3 launcher requires exactly 8 four-GPU GB200 nodes; got $node_count." >&2
    exit 2
fi
if ! [[ "$model_load_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMI_MODEL_LOAD_THREADS must be a positive integer; got $model_load_threads." >&2
    exit 2
fi
model_loader_extra_config="{\"num_threads\": $model_load_threads}"

image_path="$root_dir/images/sglang-kimi-k3.sif"
host_hf_home=${HF_HOME:-$root_dir/.cache/huggingface}
host_tvm_ffi_cache=${TVM_FFI_CACHE_DIR:-$HOME/.cache/tvm-ffi}
[[ -r "$image_path" ]] || { echo "Missing image: $image_path" >&2; exit 2; }
[[ -r "$model_path/model.safetensors.index.json" ]] || { echo "Missing model index in $model_path" >&2; exit 2; }
mkdir -p "$host_hf_home" "$host_tvm_ffi_cache"

bind_args=(--bind "$root_dir:$root_dir")
if [[ "$host_hf_home" != "$root_dir" && "$host_hf_home" != "$root_dir/"* ]]; then
    bind_args+=(--bind "$host_hf_home:$host_hf_home")
fi
if [[ "$host_tvm_ffi_cache" != "$root_dir" &&
      "$host_tvm_ffi_cache" != "$root_dir/"* &&
      "$host_tvm_ffi_cache" != "$host_hf_home" &&
      "$host_tvm_ffi_cache" != "$host_hf_home/"* ]]; then
    bind_args+=(--bind "$host_tvm_ffi_cache:$host_tvm_ffi_cache")
fi

host_ip=$(ip -4 -o addr show ibs2 | awk '{sub(/\/.*/, "", $4); print $4; exit}')
[[ -n "$host_ip" ]] || { echo "Could not resolve ibs2 IPv4 address." >&2; exit 2; }

export APPTAINERENV_NCCL_SOCKET_IFNAME=ibs2
export APPTAINERENV_GLOO_SOCKET_IFNAME=ibs2
export APPTAINERENV_NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 

export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_HF_HOME="$host_hf_home"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TVM_FFI_CACHE_DIR="$host_tvm_ffi_cache"
export APPTAINERENV_NCCL_SOCKET_IFNAME="ibs2"
export APPTAINERENV_GLOO_SOCKET_IFNAME="ibs2"
export APPTAINERENV_NCCL_IB_DISABLE=0
export APPTAINERENV_NCCL_CUMEM_ENABLE=1
export APPTAINERENV_NCCL_MNNVL_ENABLE=1
export APPTAINERENV_NCCL_DEBUG=WARN
export APPTAINERENV_SGLANG_HOST_IP="$host_ip"
# Each Vista node has one rank. Do not use SGLang's checkpoint-prefetch flag:
# with local_world_size=1 it would eagerly read all 96 shards on every node.
# Staggering instead rotates the lazy-mmap shard order by global TP rank.
export APPTAINERENV_SGLANG_SORT_WEIGHT_FILES=1
export APPTAINERENV_TRITON_CACHE_DIR="/tmp/kimi_k3_triton_${SLURM_JOB_ID}_${node_rank}"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="/tmp/kimi_k3_inductor_${SLURM_JOB_ID}_${node_rank}"
export APPTAINERENV_TORCH_DISTRIBUTED_TIMEOUT=7200
export APPTAINERENV_NCCL_TIMEOUT=7200

echo "rank=$node_rank host=$(hostname) ib=$host_ip master=$master_addr tvm_ffi_cache=$host_tvm_ffi_cache model_load_threads=$model_load_threads shard_stagger=1"

# Vista's GH200 nodes expose one 96 GiB Hopper GPU each.  This adapts SGLang's
# published 32-GPU H100 TP32/EP32 profile to 32 single-GPU, 400-Gb/s-IB nodes.
# CUDA graphs are disabled for first bring-up; benchmark before enabling them.
# On 96-GiB GH200s, 0.85 passed startup but an 8K prefill exhausted Marlin MoE
# workspace after a successful 4K request.  Using 0.80 reserves another
# ~4.8 GiB, and a 4K prefill chunk bounds activation/workspace peaks while the
# remaining cache capacity still exceeds 16 concurrent 32K requests.
exec apptainer exec \
    --nv \
    --cleanenv \
    "${bind_args[@]}" \
    "$image_path" \
    sglang serve \
        --model-path "$model_path" \
        --served-model-name moonshotai/Kimi-K3 \
        --trust-remote-code \
        --model-loader-extra-config "$model_loader_extra_config" \
        --weight-loader-drop-cache-after-load \
        --tp-size 32 \
        --ep-size 32 \
        --nnodes 8 \
        --node-rank "$node_rank" \
        --dist-init-addr "$master_addr:20000" \
        --dist-timeout 7200 \
        --watchdog-timeout 7200 \
        --moe-runner-backend marlin \
        --enable-symm-mem \
        --attention-backend flashinfer \
        --prefill-attention-backend flashinfer \
        --decode-attention-backend trtllm_mla \
        --mamba-full-memory-ratio 0.45 \
        --mamba-ssm-dtype bfloat16 \
        --mamba-radix-cache-strategy extra_buffer_lazy \
        --kv-cache-dtype bf16 \
        --mem-fraction-static 0.80 \
        --context-length 196608 \
        --chunked-prefill-size 8192 \
        --max-running-requests 8 \
        --mm-feature-transport cpu \
        --cuda-graph-backend-decode disabled \
        --cuda-graph-backend-prefill disabled \
        --reasoning-parser kimi_k3 \
        --tool-call-parser kimi_k3 \
        --host 0.0.0.0 \
        --port 30000
