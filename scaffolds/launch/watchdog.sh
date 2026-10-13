#!/usr/bin/env bash
# Source from a Slurm launcher once its output goes to the run's log; start the watchdog
# with `start_watchdog <log>` and run long steps with `run_step`.
#
# A job whose log stops growing (no progress bar, test output, or message) for the stall
# limit has stalled: when one GPU fails, for example by running out of memory, its
# data-parallel peer can wait in a collective until the time limit. The watchdog then
# logs the GPUs' state, ends the running step, and exits with STALLED_EXIT_CODE, which
# scaffolds.launch.finalize records as a stall. The limit, TRANSFORMER_STALL_MINUTES,
# must exceed the longest silent step, such as the first update's compilation.
STALLED_EXIT_CODE=124
STALL_SECONDS=$(( ${TRANSFORMER_STALL_MINUTES:-20} * 60 ))
STALL_POLL_SECONDS=60
STEP_PID=""
WATCHDOG_PID=""

# Bash defers traps until a foreground child exits, so long steps run under `wait`.
run_step() { "$@" & STEP_PID=$!; wait "$STEP_PID"; }

kill_tree() {
  local child
  for child in $(pgrep -P "$1" || true); do kill_tree "$child"; done
  kill -KILL "$1" 2>/dev/null || true
}

# Signals the launcher (USR2) once the log has not grown for the stall limit.
watch_log() {
  local log="$1" launcher="$2" size last="" idle=0
  while sleep "$STALL_POLL_SECONDS"; do
    size="$(stat -c %s "$log" 2>/dev/null || true)"
    if [[ "$size" != "$last" ]]; then last="$size"; idle=0; continue; fi
    idle=$(( idle + STALL_POLL_SECONDS ))
    if (( idle >= STALL_SECONDS )); then kill -USR2 "$launcher"; return; fi
  done
}

stop_stalled_step() {
  echo "No log output for ${STALL_SECONDS} seconds; ending stalled job ${SLURM_JOB_ID:-}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm \
    --format=csv || true
  if [[ -n "$STEP_PID" ]]; then kill_tree "$STEP_PID"; fi
  exit "$STALLED_EXIT_CODE"
}

start_watchdog() {
  trap stop_stalled_step USR2
  watch_log "$1" "$$" &
  WATCHDOG_PID=$!
}

# Call in the launcher's EXIT trap right after it saves the exit status, so a late stall
# cannot interrupt finalization.
stop_watchdog() {
  trap '' USR2
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    # Reaping it here keeps bash's job-termination notice out of the run's log.
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
}
