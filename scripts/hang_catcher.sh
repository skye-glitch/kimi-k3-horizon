#!/usr/bin/env bash
# Per-node hang catcher, started in the background by launch_k3_rank.sh.
#
# SGLang's soft watchdog (py-spy) only shows Python threads. The NCCL hangs so far
# (jobs 91870, 92084) were stuck in NCCL's proxy thread, which is a plain C thread.
# When this node's log shows a soft-watchdog dump taken while serving (its stack
# contains run_event_loop; dumps during weight loading do not), this script saves:
#   threads-<proc>.txt  every thread's state, wait channel and CPU time, sampled twice
#                       5 s apart (a thread that is spinning gains CPU time, a blocked one does not)
#   gdb-<proc>.txt      native backtraces of all threads (gdb from the SGLang image)
#   ib.txt              InfiniBand port and hardware counters that changed between two samples
#   gpu.txt             nvidia-smi utilization and memory
#   meta.txt            host, time, ptrace policy, scheduler PIDs
# into <log dir>/hang-capture/rank<N>-<capture>/.
#
# Usage: hang_catcher.sh LOG_FILE WATCH_PID OUT_DIR IMAGE NODE_RANK
# Environment: KIMI_HANG_CAPTURES (default 2), KIMI_HANG_GDB_TIMEOUT (default 180 s),
#              KIMI_HANG_GDB=image|host (default image), KIMI_HANG_POLL (default 15 s).

set -uo pipefail

log_file=$1
watch_pid=$2
out_root=$3
image_path=$4
node_rank=$5
max_captures=${KIMI_HANG_CAPTURES:-2}
gdb_timeout=${KIMI_HANG_GDB_TIMEOUT:-180}
gdb_mode=${KIMI_HANG_GDB:-image}
poll_seconds=${KIMI_HANG_POLL:-15}
min_gap_seconds=120
loop_window_seconds=120

say() { echo "[hang-catcher rank$node_rank $(date +%H:%M:%S)] $*"; }

# One line per thread: tid, name, state, utime+stime (clock ticks), last CPU, wait channel.
thread_sample() {
    local pid=$1 task stat rest name
    for task in /proc/"$pid"/task/*; do
        stat=$(cat "$task/stat" 2>/dev/null) || continue
        name=$(cat "$task/comm" 2>/dev/null)
        rest=${stat##*) }          # fields from 3 on; the name (field 2) may contain spaces
        read -r -a f <<<"$rest"
        # rest[0]=field 3 (state), rest[11]=utime, rest[12]=stime, rest[36]=processor
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${task##*/}" "$name" "${f[0]:-?}" \
            "$(( ${f[11]:-0} + ${f[12]:-0} ))" "${f[36]:-?}" "$(cat "$task/wchan" 2>/dev/null || echo ?)"
    done
}

# The HCAs NCCL uses (launch_k3_rank.sh sets APPTAINERENV_NCCL_IB_HCA), else every mlx5 device.
hca_list=${APPTAINERENV_NCCL_IB_HCA:-}
hca_list=${hca_list//,/ }
if [[ -z "${hca_list// /}" ]]; then
    hca_list=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')
fi

ib_sample() {
    local hca file
    for hca in $hca_list; do
        for file in /sys/class/infiniband/"$hca"/ports/*/counters/* /sys/class/infiniband/"$hca"/ports/*/hw_counters/*; do
            [[ -r "$file" ]] && printf '%s %s\n' "${file#/sys/class/infiniband/}" "$(cat "$file" 2>/dev/null)"
        done
    done
}

# File-safe process name from the first argument: sglang::scheduler_TP11_EP11 -> scheduler_TP11_EP11.
proc_name() {
    local name
    name=$(head -z -n1 /proc/"$1"/cmdline 2>/dev/null | tr -d '\0')
    name=${name#sglang::}
    name=${name//[^A-Za-z0-9_.-]/_}
    echo "${name:-$1}"
}

capture() {
    local n=$1 dir pids pid name a b
    dir="$out_root/rank${node_rank}-${n}"
    mkdir -p "$dir"
    pids=$(pgrep -u "$(id -u)" -f '^sglang::scheduler' || true)
    {
        echo "host=$(hostname) time=$(date --iso-8601=seconds) capture=$n"
        echo "ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo n/a)"
        for pid in $pids; do echo "pid=$pid $(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null)"; done
    } >"$dir/meta.txt"
    if [[ -z "$pids" ]]; then
        say "no sglang::scheduler processes found; nothing captured"
        return
    fi

    a=$(mktemp -d)
    for pid in $pids; do thread_sample "$pid" >"$a/$pid"; done
    ib_sample >"$a/ib"
    sleep 5
    for pid in $pids; do
        name=$(proc_name "$pid")
        thread_sample "$pid" >"$a/$pid.b"
        # Join the two samples by tid and add the CPU-tick delta; busiest threads first.
        {
            printf 'tid\tname\tstate\tcpu_ticks_in_5s\tlast_cpu\twchan\n'
            awk -F'\t' 'NR==FNR {t[$1]=$4; next} {d = ($1 in t) ? $4 - t[$1] : "new"; printf "%s\t%s\t%s\t%s\t%s\t%s\n", $1, $2, $3, d, $5, $6}' \
                "$a/$pid" "$a/$pid.b" | sort -t$'\t' -k4,4nr
        } >"$dir/threads-$name.txt"
    done
    ib_sample >"$a/ib.b"
    {
        echo "# counters that changed during 5 s (name before after delta)"
        awk 'NR==FNR {v[$1]=$2; next} ($1 in v) && $2 != v[$1] {print $1, v[$1], $2, $2 - v[$1]}' "$a/ib" "$a/ib.b"
        echo "# all counters at the second sample"
        cat "$a/ib.b"
    } >"$dir/ib.txt"
    rm -rf "$a"
    nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used,power.draw --format=csv >"$dir/gpu.txt" 2>&1 || true

    # Native stacks, all schedulers in parallel. gdb pauses each process while it reads.
    for pid in $pids; do
        name=$(proc_name "$pid")
        gdb_args=(-p "$pid" -batch -nx -ex "set pagination off" -ex "set print thread-events off" -ex "thread apply all bt 40")
        if [[ "$gdb_mode" == image ]]; then
            timeout "$gdb_timeout" apptainer exec --nv --cleanenv "$image_path" gdb "${gdb_args[@]}" >"$dir/gdb-$name.txt" 2>&1 &
        else
            timeout "$gdb_timeout" gdb -ex "set sysroot /proc/$pid/root" "${gdb_args[@]}" >"$dir/gdb-$name.txt" 2>&1 &
        fi
    done
    wait
    say "capture $n written to $dir"
}

# True if the log bytes appended since the last poll contain $1.
chunk_has() { tail -c +"$((offset + 1))" "$log_file" | head -c "$((size - offset))" | grep -q -F "$1"; }

say "armed: watching $log_file (up to $max_captures captures, gdb from $gdb_mode)"
offset=0
captures=0
last_capture=0
last_loop_seen=0
while kill -0 "$watch_pid" 2>/dev/null && (( captures < max_captures )); do
    sleep "$poll_seconds"
    size=$(stat -c %s "$log_file" 2>/dev/null || echo 0)
    (( size > offset )) || continue
    now=$(date +%s)
    chunk_has "run_event_loop" && last_loop_seen=$now
    if chunk_has "Scheduler watchdog timeout" &&
        (( now - last_loop_seen <= loop_window_seconds )) &&
        (( now - last_capture >= min_gap_seconds )); then
        captures=$((captures + 1))
        last_capture=$now
        say "soft watchdog fired while serving; capturing ($captures/$max_captures)"
        capture "$captures"
    fi
    offset=$size
done
