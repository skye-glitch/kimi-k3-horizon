# Kimi K3 on TACC Horizon

This repository serves `moonshotai/Kimi-K3` with SGLang across Horizon nodes. 
The production checkpoint, multi-node startup, generation, 32K context, GPU
monitoring, and multi-turn chat path have been validated.

Detailed bring-up history, failed experiments, measurements, and design choices
are kept in [the bring-up report](reports/kimi-k3-bringup-report.md). This README
is the operational runbook.

## Repository layout

| Path | Purpose |
|---|---|
| `images/sglang-kimi-k3-cu12-74968e5653-arm64.sif` | ARM64 SGLang runtime image |
| `models/Kimi-K3/` | Production checkpoint (96 shards, about 1.56 TB decimal) |
| `slurm/download_k3.sbatch` | Resumable checkpoint download on a CPU node |
| `slurm/serve_k3_32gh.sbatch` | Legacy 32-node serving job |
| `slurm/server_k3_auto_kimi.sbatch` | Production 8-node job for automated Engram evaluation |
| `scripts/preflight.sh` | Validate the image, index, and all checkpoint shards |
| `scripts/chat.py`, `chat.sh` | Streaming multi-turn client and convenience wrapper |
| `scripts/context_probe.py` | Exact-token long-context recall test |
| `scripts/summarize_context_gpu.py` | Join context results with GPU samples |
| `logs/JOBID/` | All outputs and CSV files for one Slurm job |

The runtime image, checkpoint, caches, and logs are local artifacts excluded
from Git.

## Requirements and current profile

Run Slurm commands from a Horizon login node. The scripts load the required
modules themselves. For direct Python, GPU, virtual-environment, or container
work, load the modules in the same shell before use:

```bash
module load gcc
module load cuda
module load tacc-apptainer

Do not add Horizon's host `nccl` or `nvshmem` modules to the serving jobs. The
container supplies the matching CUDA/NCCL stack and is launched with
`--cleanenv`.

The production launcher currently uses:

| Setting | Value |
|---|---:|
| Nodes / GPUs | 8 nodes, one GPU per task |
| Tensor / expert parallelism | TP32 / EP32 |
| Context limit | 32,768 tokens |
| Maximum running requests | 16 |
| Chunked prefill | 4,096 tokens |
| Static memory fraction | 0.80 |
| KV cache | FP8 E4M3 |
| KDA/Mamba state | BF16 |
| CUDA graphs | Disabled |
| API | OpenAI-compatible, rank 0 port 30000 |

Change one memory/performance setting at a time and remeasure before combining
changes.

## Instructions

### 1. Prepare the image and checkpoint

Clone the repository on Horizon, enter its root directory, and obtain the runtime
image if it is not already present:

```bash
cd /path/to/kimi_k3
mkdir -p images
module load tacc-apptainer
apptainer pull \
  images/sglang-kimi-k3-cu12-74968e5653-arm64.sif \
  docker://lmsysorg/sglang:kimi-k3-cu12-74968e5653-arm64
```

Validate an existing checkpoint before an expensive run:

```bash
bash scripts/preflight.sh
```

If the checkpoint is missing or has been purged, submit the resumable download:

```bash
export HF_HOME=/path/to/your/huggingface/cache
export HF_TOKEN=hf_your_token
sbatch -A PROJECT_ID slurm/download_k3.sbatch
```

The token is optional for this public model but helps with rate limits. Do not
write it into a script. `HF_HOME` stores Hugging Face cache data; the deployable
checkpoint is intentionally written to `models/Kimi-K3/`.

Monitor the download with:

```bash
squeue -u "$USER"
tail -f logs/JOBID/k3-download.out
```

### 2. Automated Evaluation with Engram

To run the full end-to-end evaluation pipeline, submit the automated Slurm script:

```bash
sbatch slurm/server_k3_auto_kimi.sbatch
```

This 8-node script (configured for the debug partition) is designed to streamline evaluation. It automatically launches the distributed Kimi SGLang server and, once the endpoints are healthy, seamlessly connects the server to the Kimi evaluation scripts located in the Engram directory. All subsequent benchmarking and generation tasks are handled sequentially within the job allocation, minimizing manual intervention and ensuring accurate metrics capture.

### 3.Start a multi-turn conversation

From the Horizon project directory, pass only the serving Job ID:

```bash
./chat.sh JOBID
```

`chat.sh` obtains the rank-0 hostname directly from Slurm's batch-host field
(`squeue -o '%B'`). The serving job must be running. To allow a longer single
response than the default 1024 tokens, add—for example—`--max-tokens 4096`.

The client streams K3's reasoning and answer as they arrive. Commands are:

- `/clear`: clear the client-side conversation history;
- `/exit`: close the client without stopping the server.

`/clear` does not flush the server-wide SGLang cache. The client preserves and
resends complete assistant messages, including `reasoning_content` and tool
calls, as required for K3 multi-turn conversations.

Each turn is appended to `logs/JOBID/k3-chat.csv`, including client-observed
TTFT, time to first answer-content token, total latency, token usage, HTTP
status, and timestamps.

### 4. Connect from a workstation (optional)

Do not expose port 30000 publicly. Create an SSH tunnel using the rank-0 host:

```bash
ssh -L 30000:RANK0_HOST:30000 TACC_USERNAME@Horizon.tacc.utexas.edu
```

Then address `http://127.0.0.1:30000/v1` from a local OpenAI-compatible client.
For this repository's client:

```bash
python3 scripts/client.py \
  --base-url http://127.0.0.1:30000/v1 \
  --prompt 'Reply with exactly: ready'
```

Running `chat.sh` on Horizon is preferred when its CSV must be written directly
into the shared per-job log directory.



### 5. Monitor and stop the job

Every serving job samples all 32 GPUs every 15 seconds. The interval can be
changed at submission with `KIMI_GPU_MONITOR_INTERVAL`.

```bash
tail -f logs/JOBID/k3-gpu-monitor.out
tail -f logs/JOBID/k3-rank-0.out
scancel JOBID
sacct -j JOBID -X -o JobID,State,Elapsed,AllocNodes,Start,End
```

Cancel the allocation when it is no longer needed; idle servers are still
charged for all 8 nodes.

## Logs

When jobs are submitted from the repository root, all project-generated output
is grouped by Slurm Job ID:

```text
logs/JOBID/
├── k3-server.out
├── k3-rank-0.out ... k3-rank-8.out
├── k3-gpu-monitor.out
├── k3-gpu.csv
├── k3-chat.csv                 # after interactive chat
├── k3-context.csv              # after a context probe
└── k3-context-gpu-summary.csv  # after summarization
```

Download jobs follow the same directory rule with a workload-specific filename.

## Troubleshooting

| Symptom | Check or action |
|---|---|
| Server still starting | Read `k3-server.out` and `k3-rank-0.out`; wait for the Hello PASS marker |
| Client waits but rank 0 shows no POST | Verify the rank-0 hostname, base URL, port, and SSH tunnel |
| HTTP 400 context-length error | Reduce prompt history or `--max-tokens` |
| GPU memory remains high after a request | Finished radix/Mamba/CUDA cache can remain reserved; inspect trends rather than one sample |
| OOM or Marlin workspace failure | Keep the validated 0.80 memory fraction and 4K prefill chunk; avoid retained-cache stress until idle |
| One rank exits | Inspect the matching `k3-rank-RANK.out`; `srun --kill-on-bad-exit` will stop the distributed step |

Horizon provides one GH200 per node and no inter-GPU NVLink between these nodes.
Autoregressive generation therefore communicates over InfiniBand and has much
higher latency than a multi-GPU NVSwitch server.

`$SCRATCH` is purgeable. Maintain a retention or re-download plan for the 1.56
TB checkpoint. Review the Kimi K3 license before public or commercial serving.

## References

- [Moonshot Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Moonshot Kimi K3 repository and license](https://github.com/MoonshotAI/Kimi-K3)
- [SGLang Kimi K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- [TACC Horizon documentation](https://docs.tacc.utexas.edu/hpc/Horizon/)
