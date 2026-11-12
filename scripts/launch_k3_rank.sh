#!/usr/bin/env bash


set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=${KIMI_K3_ROOT:-$(cd "$script_dir/.." && pwd)}
model_path=${KIMI_MODEL_PATH:-$root_dir/models/Kimi-K3}
master_addr=${MASTER_ADDR:?MASTER_ADDR must be the rank-0 InfiniBand address}
node_rank=${SLURM_PROCID:?This script must run under srun}
node_count=${SLURM_NNODES:?This script must run inside a Slurm job}
model_load_threads=${KIMI_MODEL_LOAD_THREADS:-2}

# Each GB200 node has 4 GPUs (~182 GB usable each). The MXFP4 checkpoint needs
# ~58.6 GB/GPU at TP32, ~117 GB/GPU at TP16; TP8 (2 nodes, ~234 GB/GPU) cannot fit.
# 96 attention heads and 896 experts divide evenly by both 16 and 32.
gpus_per_node=4
if [[ "$node_count" -ne 4 && "$node_count" -ne 8 ]]; then
    echo "The K3 launcher supports 4 nodes (TP16/EP16) or 8 nodes (TP32/EP32) of four-GPU GB200; got $node_count." >&2
    exit 2
fi
tp_size=$((node_count * gpus_per_node))
if ! [[ "$model_load_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMI_MODEL_LOAD_THREADS must be a positive integer; got $model_load_threads." >&2
    exit 2
fi
model_loader_extra_config="{\"num_threads\": $model_load_threads}"

image_path="$root_dir/images/sglang-kimi-k3.sif"
host_hf_home=${HF_HOME:-$root_dir/.cache/huggingface}
# Apptainer mounts $HOME into the container, so any cache left at its default
# (~/.cache/*, ~/.nv, ~/.triton) lands on the small $HOME quota. Keep them on scratch.
cache_root=${KIMI_CACHE_ROOT:-/scratch/07980/sli4/apptainer_cache}
host_tvm_ffi_cache=${TVM_FFI_CACHE_DIR:-$cache_root/tvm-ffi}
[[ -r "$image_path" ]] || { echo "Missing image: $image_path" >&2; exit 2; }
[[ -r "$model_path/model.safetensors.index.json" ]] || { echo "Missing model index in $model_path" >&2; exit 2; }
mkdir -p "$host_hf_home" "$host_tvm_ffi_cache" "$cache_root/xdg" "$cache_root/nv/ComputeCache"

bind_args=(--bind "$root_dir:$root_dir" --bind "$cache_root:$cache_root")
if [[ "$host_hf_home" != "$root_dir" && "$host_hf_home" != "$root_dir/"* ]]; then
    bind_args+=(--bind "$host_hf_home:$host_hf_home")
fi
if [[ "$host_tvm_ffi_cache" != "$root_dir" &&
      "$host_tvm_ffi_cache" != "$root_dir/"* &&
      "$host_tvm_ffi_cache" != "$cache_root/"* &&
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
export APPTAINERENV_XDG_CACHE_HOME="$cache_root/xdg"
# FlashInfer JIT cache is $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer (default base: $HOME).
export APPTAINERENV_FLASHINFER_WORKSPACE_BASE="$cache_root"
export APPTAINERENV_CUDA_CACHE_PATH="$cache_root/nv/ComputeCache"
export APPTAINERENV_NCCL_SOCKET_IFNAME="ibs2"
export APPTAINERENV_GLOO_SOCKET_IFNAME="ibs2"
export APPTAINERENV_NCCL_IB_DISABLE=0
export APPTAINERENV_NCCL_CUMEM_ENABLE=1
export APPTAINERENV_NCCL_MNNVL_ENABLE=1
# Debugging the post-startup hang (jobs 85143, 88152): verbose NCCL logs plus the
# PyTorch NCCL flight recorder, dumped into the job log dir on an NCCL timeout.
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export APPTAINERENV_TORCH_NCCL_DUMP_ON_TIMEOUT=1
export APPTAINERENV_TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${KIMI_JOB_LOG_DIR:-/tmp}/nccl_trace_rank${node_rank}_"
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
        --tp-size "$tp_size" \
        --ep-size "$tp_size" \
        --nnodes "$node_count" \
        --node-rank "$node_rank" \
        --dist-init-addr "$master_addr:20000" \
        --dist-timeout 7200 \
        --watchdog-timeout 300 \
        --moe-runner-backend marlin \
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
