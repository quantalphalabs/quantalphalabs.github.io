#!/usr/bin/env sh
# Downloads the public nflverse files sim.py reads into the directory given (default ./data).
set -eu
dir="${1:-data}"
mkdir -p "$dir"
base="https://github.com/nflverse/nflverse-data/releases/download"
for f in pbp/play_by_play_2025.parquet pbp/play_by_play_2026.parquet \
         ftn_charting/ftn_charting_2025.parquet ftn_charting/ftn_charting_2026.parquet \
         rosters/roster_2025.parquet rosters/roster_2026.parquet \
         injuries/injuries_2026.parquet schedules/games.csv; do
  curl -fsSL -o "$dir/$(basename "$f")" "$base/$f"
  echo "fetched $f"
done
