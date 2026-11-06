#!/usr/bin/env bash
# Low-frequency, read-only watch of ~/Downloads: every INTERVAL seconds append the size of each in-progress
# (.crdownload) file and of the named dependency files to LOG. A download is complete once its final name exists
# and two consecutive entries show the same size.
INTERVAL=${INTERVAL:-600}
LOG=${LOG:-$(dirname "$0")/../develop_logs/test_runs/20261107_evalkit/downloads.log}
mkdir -p "$(dirname "$LOG")"
while true; do
  {
    echo "== $(date -Iseconds)"
    find "$HOME/Downloads" -maxdepth 1 -newermt '2026-11-06' -type f -printf '%s\t%TY-%Tm-%Td_%TH:%TM\t%f\n' | sort -k3
  } >> "$LOG"
  sleep "$INTERVAL"
done
