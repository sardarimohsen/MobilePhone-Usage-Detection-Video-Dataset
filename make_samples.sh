#!/bin/bash
# usage: bash make_samples.sh videos.txt [output_dir]
#
# videos.txt: one video per line, path relative to the Dataset/Dataset folder, e.g.
#   Faculty Parking/Faculty Parking_2.mp4
#
# For each video it copies
#   - the video                       -> <output_dir>/videos/<same relative path>
#   - its annotation folder files     -> <output_dir>/annotations/<scene>/<video name>/
#     (only .xml .json .csv .yaml; the duplicate mp4, jpg, sqlite and bak files are skipped)
set -u
V="${V_ROOT:-/home/meybod/vault-open/Dataset/Dataset}"
A="${A_ROOT:-/home/meybod/vault-open/Dataset_annotations}"
LIST="$1"
OUT="$(realpath -m "${2:-samples}")"
mkdir -p "$OUT/videos" "$OUT/annotations"
while IFS= read -r r || [ -n "$r" ]; do
  r="${r#./}"; [ -z "$r" ] && continue
  if [ ! -f "$V/$r" ]; then echo "MISSING VIDEO: $r"; continue; fi
  [ -f "$OUT/videos/$r" ] || (cd "$V" && cp --parents -- "$r" "$OUT/videos/")
  d="${r%.*}"                      # <scene>/<video name>
  if [ ! -d "$A/$d" ]; then echo "NO ANNOTATION FOLDER: $A/$d"; continue; fi
  n=0
  while IFS= read -r -d '' f; do
    rel="${f#$A/}"
    (cd "$A" && cp --parents -- "$rel" "$OUT/annotations/")
    n=$((n+1))
  done < <(find "$A/$d" -type f \( -name "*.xml" -o -name "*.json" -o -name "*.csv" -o -name "*.yaml" \) -print0)
  echo "ok: $r  (+$n annotation files)"
done < "$LIST"
du -sh "$OUT/videos" "$OUT/annotations"
