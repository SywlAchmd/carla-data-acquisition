#!/usr/bin/env bash
# Supervisor: keeps the CARLA server and the capture alive until TOTAL frames exist.
# The server occasionally segfaults on a 4 GB card (or if a sync-mode client is
# killed mid-tick); this restarts it and resumes where the capture left off.
#
#   ./run_capture.sh 5000 out
set -u
TOTAL=${1:-5000}
OUT=${2:-out}
TRAFFIC=${TRAFFIC:-12}
CARLA_ROOT=${CARLA_ROOT:-$HOME/Sawal/CARLA_0.9.16}
LOG=${LOG:-$PWD/capture.log}
SERVER_LOG=${SERVER_LOG:-$PWD/carla_server.log}

count_frames() { find "$OUT/raw" -name '*.jpg' 2>/dev/null | wc -l; }

start_server() {
  pkill -f CarlaUE4-Linux-Shipping 2>/dev/null
  sleep 5
  ( cd "$CARLA_ROOT" && nohup ./CarlaUE4.sh -quality-level=Low -RenderOffScreen \
      -nosound -carla-rpc-port=2000 >>"$SERVER_LOG" 2>&1 & )
  for _ in $(seq 1 40); do
    if python3 -c "
import carla,sys
c=carla.Client('127.0.0.1',2000); c.set_timeout(8.0)
try: c.get_world().get_map(); sys.exit(0)
except Exception: sys.exit(1)" 2>/dev/null; then echo "server ready"; return 0; fi
    sleep 5
  done
  echo "server did not come up"; return 1
}

for attempt in $(seq 1 12); do
  have=$(count_frames)
  [ "$have" -ge "$TOTAL" ] && { echo "done: $have frames"; exit 0; }
  echo "=== attempt $attempt: $have/$TOTAL frames ==="
  start_server || continue
  python3 -u capture_overtaking.py --total-frames "$TOTAL" --out "$OUT" --resume \
      --max-frames-per-run 320 --traffic "$TRAFFIC" --seed 2024 2>&1 | tee -a "$LOG"
done
echo "gave up after 12 attempts with $(count_frames)/$TOTAL frames"
exit 1
