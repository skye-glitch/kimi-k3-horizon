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
| `slurm/server_k3_auto_kimi.sbatch` | Serving job (4 nodes by default, or 8) that also runs the Engram Kimi benchmark |
| `scripts/launch_k3_rank.sh` | Per-node SGLang launcher that `srun` starts on every node |
| `scripts/hang_catcher.sh` | Per-node hang catcher the launcher starts: native thread stacks and InfiniBand counters when a rank stalls |
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
```

Do not add Horizon's host `nccl` or `nvshmem` modules to the serving jobs. The
container supplies the matching CUDA/NCCL stack and is launched with
`--cleanenv`.

The launcher (`scripts/launch_k3_rank.sh`) currently uses:

| Setting | Value |
|---|---:|
| Nodes / GPUs | 4 nodes (default) or 8 nodes of GB200 NVL4; one task per node, 4 Blackwell GPUs (185 GiB HBM3) each |
| Tensor / expert parallelism | TP16 / EP16 on 4 nodes, TP32 / EP32 on 8 nodes |
| Context limit | 196,608 tokens |
| Maximum running requests | 8 |
| Chunked prefill | 8,192 tokens |
| Static memory fraction | 0.80 |
| KV cache | BF16 |
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

### 2. Launch the server and the Engram benchmark

`slurm/server_k3_auto_kimi.sbatch` starts the K3 server and, once it is healthy,
runs the Engram Kimi benchmark (PageIndex vs. LightRAG) inside the same job.

**Before the first run**

- **Engram checkout.** The job runs the benchmark from the Engram path set near the
  end of the sbatch script (`engram_dir=/scratch/07980/sli4/Engram`). Edit that
  line if your checkout is elsewhere.
  That checkout needs its environment installed (`uv sync`), a `.env` with the
  Tejas embedding settings, and the benchmark data. See
  `playground/graph_rag/README.md` in Engram.
- **Caches.** The job keeps the FlashInfer, tvm-ffi, CUDA, Triton and XDG caches
  in `KIMI_CACHE_ROOT` (default `/scratch/07980/sli4/apptainer_cache`) instead of
  `$HOME`, which has a small quota. Set it to your own scratch directory:
  `export KIMI_CACHE_ROOT=$SCRATCH/apptainer_cache` before `sbatch`.
- **Account and email.** Change `#SBATCH -A`, `--mail-user` and `--mail-type` in
  the script to your allocation and address.

**Submit**

Run `sbatch` from the repository root. The job uses its submit directory as the
repository root, so the model, image and logs paths depend on it.

```bash
cd /path/to/kimi-k3-vista
sbatch slurm/server_k3_auto_kimi.sbatch
```

The defaults are 4 nodes (TP16/EP16) for 1 hour in the `debug` partition. Values
given to `sbatch` on the command line override the `#SBATCH` lines:

```bash
sbatch --time=04:00:00 slurm/server_k3_auto_kimi.sbatch          # longer run
sbatch --nodes=8 --ntasks=8 slurm/server_k3_auto_kimi.sbatch     # TP32/EP32
sbatch -A PROJECT_ID slurm/server_k3_auto_kimi.sbatch            # other allocation
```

Only 4 or 8 nodes are accepted: TP8 on 2 nodes does not fit in GPU memory. To
raise the time of a job that is still pending without losing its place in the
queue, use `scontrol update JobId=JOBID TimeLimit=04:00:00`.

**Choosing a time limit**

- Job priority on Horizon does not depend on the requested time. Before choosing,
  compare start estimates with
  `sbatch --test-only --time=HH:MM:SS slurm/server_k3_auto_kimi.sbatch`.
- Startup takes about 11 minutes on 4 nodes, so longer jobs leave more of their
  time for the benchmark.
- The server stays up until the time limit, even after the benchmark finishes,
  and the whole allocation is charged.
- A job that would overlap a maintenance reservation does not start. Close to a
  maintenance window, add `--time-min=00:15:00` so Slurm can shorten the job to
  fit.

**What the job does**

1. Runs `scripts/preflight.sh`, then starts SGLang on every node through `srun`.
2. Waits for `/health_generate` (`KIMI_STARTUP_TIMEOUT`, default 7200 s), then
   sends a Hello request (`KIMI_HELLO_TIMEOUT`, default 1800 s). When it passes,
   `k3-server.out` prints `K3_HELLO_INFERENCE_TEST=PASS`.
3. Starts Engram's `playground/graph_rag/run_evaluation_kimi.sh JOBID` in the
   background. The benchmark checkpoints its progress, so the next job resumes
   where this one stopped.
4. If the server step exits early (a crash, or the watchdog after a hang), stops
   the benchmark and adds one line saying so to `evaluation_kimi.log`. The job
   then ends, which frees the nodes.

**Follow progress**

```bash
squeue -u "$USER"
tail -f logs/JOBID/k3-server.out                 # startup and Hello test
tail -f logs/JOBID/k3-rank-0.out                 # SGLang rank 0 and requests
tail -f /path/to/Engram/log/evaluation_kimi.log  # benchmark scores
tail -f /path/to/Engram/log/api_graph_kimi.log   # LightRAG ingest
```

### 3. Start a multi-turn conversation

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

Every serving job samples all of its GPUs every 15 seconds. The interval can be
changed at submission with `KIMI_GPU_MONITOR_INTERVAL`.

```bash
tail -f logs/JOBID/k3-gpu-monitor.out
tail -f logs/JOBID/k3-rank-0.out
scancel JOBID
sacct -j JOBID -X -o JobID,State,Elapsed,AllocNodes,Start,End
```

Cancel the allocation when it is no longer needed; idle servers are still
charged for every node in the job.

## Logs

When jobs are submitted from the repository root, all project-generated output
is grouped by Slurm Job ID:

```text
logs/JOBID/
├── k3-server.out
├── k3-rank-0.out ... k3-rank-N.out   # one per node
├── nccl_trace_rank*                # only after an NCCL timeout
├── hang-capture/rankN-M/          # only if the soft watchdog fires while serving
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
| OOM or Marlin workspace failure | Keep the 0.80 memory fraction and 8K prefill chunk the launcher uses; avoid retained-cache stress until idle |
| One rank exits | Inspect the matching `k3-rank-RANK.out`; `srun --kill-on-bad-exit` will stop the distributed step |
| Server stops making progress 75 minutes to about 2 h 20 min after start (an NCCL all-reduce hangs) | Not yet fixed. In jobs 91870 and 92084, TP11's CPU was stuck in `ncclLocalOpAppend`: NCCL's proxy thread had stopped draining its queue, and the other ranks waited for it. Three sources of evidence: (1) `grep -n "Pyspy dump" logs/JOBID/k3-rank-*.out` shows each rank's Python stack (dumps during weight loading at startup are normal); (2) `logs/JOBID/nccl_trace_rank*` are PyTorch flight-recorder pickles, and the rank whose `last_enqueued_collective` is lowest is the one that fell behind; (3) `logs/JOBID/hang-capture/rankN-M/` comes from `scripts/hang_catcher.sh`, which, when the soft watchdog fires while serving, saves every thread's state, CPU time over 5 s and native gdb stack (NCCL's proxy thread included), plus InfiniBand counter changes and `nvidia-smi`. Timeouts (240/600/900 s by default): `KIMI_SOFT_WATCHDOG_TIMEOUT`, `KIMI_DIST_TIMEOUT`, `KIMI_WATCHDOG_TIMEOUT`. `KIMI_HANG_CATCHER=0` turns the catcher off. Lines longer than 4,000 characters in `k3-rank-*.out` (the watchdog's `leaked_full_pages` sets and request `input_ids`) are cut, with a note saying how much; `KIMI_LOG_LINE_MAX` changes the limit |

Horizon's Grace Blackwell nodes use the GB200 NVL4 platform. Each node has 4
Blackwell GPUs with 185 GiB of HBM3 each, and 2 Grace CPUs with 478 GiB of LPDDR5X
and 72 cores each. NVLink connects the 4 GPUs inside a node, and nodes talk to each
other over InfiniBand (`ibs2`). TP16 spans 4 nodes and TP32 spans 8, so every
decode step's collectives cross InfiniBand. Expect much higher per-token latency
than on a single NVSwitch server.

`$SCRATCH` is purgeable. Maintain a retention or re-download plan for the 1.56
TB checkpoint. Review the Kimi K3 license before public or commercial serving.

## References

- [Moonshot Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Moonshot Kimi K3 repository and license](https://github.com/MoonshotAI/Kimi-K3)
- [SGLang Kimi K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- [TACC Horizon documentation](https://docs.tacc.utexas.edu/hpc/Horizon/)
