#!/usr/bin/env bash
# Memory watchdog: run any command under it; the command is killed before the
# box can thrash.
#
#   bench/memguard.sh -- <command...>
#
# Exists because of 2026-08-04 ~07:15: a --novram benchmark run put ~37 GB of
# model weights into 31 GB of host RAM, kswapd went into a stall loop
# (journalctl -b -1: folio_alloc_swap / kswapd traces, swap filling), the
# desktop froze and the machine needed a hard reset.
#
# Kill rules (checked every 2s):
#
#   MemAvailable < 150 MB                  — absolute backstop, fire
#   MemAvailable < 400 MB for 30s straight — sustained starvation, not a dip
#   swap +> 6 GB total over the run        — far past anything normal here
#   5 unreadable /proc/meminfo samples     — cannot even fork; that IS distress
#
# THE MECHANISM, NOT THE SYMPTOMS — and on this box almost every symptom is
# also normal. Measured during HEALTHY runs on 2026-08-04:
#
#   MemAvailable 515-2205 MB   staging a 15 GB encoder + 20 GB DiT into 31 GB,
#                              and again at every VAE decode. Brief, recovers.
#   swap +173 to +3577 MB      the same staging. zram is COMPRESSED RAM and
#                              growth is how it works.
#   swap rate 2133 MB per 8s   observed with 14 GB free and only +221 MB net:
#                              zram churns pages in and out, so instantaneous
#                              RATE oscillates and distinguishes nothing.
#
# Rate was therefore tried and abandoned. What separates normal from the freeze
# is DURATION and TOTAL: the freeze was +4 GB in 20s and STILL CLIMBING with
# memory exhausted and staying exhausted, so it crosses both the 6 GB total and
# the 30s starvation rule, while a healthy staging burst crosses neither.
#
# Calibration history, every step paid for in a dead run on 2026-08-04:
#   2 GB single-sample floor  -> killed a healthy phase-K run 33 MiB short
#   1.2 GB hard floor         -> killed K11 at its VAE decode, swap +173 MiB
#   700 MiB critical floor    -> killed Q1 at 515 MiB with swap FALLING 1685 MiB
#   bare 2 GB swap-growth cap -> killed Q1 at swap +2583 MiB with 12110 MiB free
#   low-AND-paging, 2 ticks   -> killed D0 mid-staging at 1505 MiB / +3357 MiB
#   1.5 GB per 8s rate rule   -> killed D0 again on a staging burst, then a row
#                                with 14 GB free on pure zram churn
# Eight kills, seven of them wrong. Every wrong one came from treating an
# instantaneous reading as evidence, when normal operation here produces those
# same readings several times a minute.
#
# NOTE an ordinary OOM kill is NOT the failure being prevented: the kernel reaps
# the process and the box carries on, which for a benchmark is a failed row.
# Only thrash-to-frozen-desktop costs a hard reset.
#
# On trip: SIGKILL the whole process group. A dead benchmark row is nothing; a
# frozen machine loses every unsaved thing on it.
set -uo pipefail

AVAIL_FLOOR_MIB=150       # about to be OOM-killed regardless of anything else
AVAIL_LOW_MIB=400         # "starving", but only when it STAYS there
AVAIL_LOW_TICKS_MAX=15    # 15 samples x 2s = 30s of continuous starvation
SWAP_GROW_MAX_MIB=6144    # total growth; healthy runs here top out near 3.6 GB
BAD_READS_MAX=5           # 10s of being unable to fork awk is itself distress

[[ "${1:-}" == "--" ]] && shift
[[ $# -ge 1 ]] || { echo "usage: memguard.sh -- <command...>"; exit 64; }

mem_avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null; }
swap_used() { awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo 2>/dev/null; }

# A sample is usable only if BOTH readings parsed as integers. Not paranoia:
# each reading forks awk, and fork is exactly what starts failing under the
# pressure this script watches for. An empty result reaches bash arithmetic,
# which evaluates "" as 0, trips the floor, and kills a healthy run on a reading
# that never happened — the log said `MemAvailable=0MiB ... swap+-2427MiB`, an
# impossible pair that was simply one failed fork.
usable() { [[ "$1" =~ ^-?[0-9]+$ && "$2" =~ ^-?[0-9]+$ ]]; }

die() {
  echo "[memguard] TRIPPED: $* — killing pgid $pgid" >&2
  kill -KILL -- "-$pgid" 2>/dev/null
  # also take down any comfy servers the command spawned detached
  for p in $(pgrep -f "ma""in.py --po""rt"); do kill -KILL "$p" 2>/dev/null; done
  exit 75
}

swap0=$(swap_used)
setsid "$@" &
child=$!
pgid=$(ps -o pgid= "$child" | tr -d ' ')

trap 'kill -TERM -- "-$pgid" 2>/dev/null' INT TERM

low_ticks=0
bad_reads=0
while kill -0 "$child" 2>/dev/null; do
  sleep 2
  avail=$(mem_avail); sw=$(swap_used)

  if ! usable "$avail" "$sw"; then
    bad_reads=$((bad_reads + 1))
    (( bad_reads >= BAD_READS_MAX )) && \
      die "${bad_reads} consecutive unreadable /proc/meminfo samples (cannot fork)"
    continue
  fi
  bad_reads=0
  grown=$((sw - swap0))

  if (( avail < AVAIL_LOW_MIB )); then
    low_ticks=$((low_ticks + 1))
  else
    low_ticks=0
  fi

  (( avail < AVAIL_FLOOR_MIB )) && \
    die "MemAvailable=${avail}MiB below floor (swap+${grown}MiB)"
  (( low_ticks >= AVAIL_LOW_TICKS_MAX )) && \
    die "MemAvailable=${avail}MiB for $((low_ticks * 2))s straight (swap+${grown}MiB)"
  (( grown > SWAP_GROW_MAX_MIB )) && \
    die "swap+${grown}MiB total (MemAvailable=${avail}MiB)"
done
wait "$child"
