#!/usr/bin/env bash

set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$ROOT_DIR/inputs"
RESULTS_DIR="$ROOT_DIR/results/$(date +%Y%m%d-%H%M%S)"

THREADS="$(nproc --all)"
ITERS="100000"
BATCH="1024"
BLOCK="256"

IMPLEMENTATIONS=("seqMDS" "parMDS" "gpuMDS" "gpucpuMDS" "gpucpuMDS_v2")
REPRESENTATIVE_INPUTS=(
  "Antwerp1.vrp"
  "Golden_12.vrp"
  "CMT5.vrp"
  "X-n1001-k43.vrp"
)

mkdir -p "$RESULTS_DIR"

make -C "$ROOT_DIR"

run_impl() {
  local impl="$1"
  local input_path="$2"
  local round_flag="$3"
  local input_name
  input_name="$(basename "$input_path" .vrp)"
  local stdout_file="$RESULTS_DIR/${impl}-${input_name}.stdout.log"
  local stderr_file="$RESULTS_DIR/${impl}-${input_name}.stderr.log"
  local status_file="$RESULTS_DIR/${impl}-${input_name}.status"

  case "$impl" in
    seqMDS)
      "$ROOT_DIR/seqMDS.out" "$input_path" -round "$round_flag" >"$stdout_file" 2>"$stderr_file"
      ;;
    parMDS)
      "$ROOT_DIR/parMDS.out" "$input_path" -nthreads "$THREADS" -round "$round_flag" >"$stdout_file" 2>"$stderr_file"
      ;;
    gpuMDS)
      "$ROOT_DIR/gpuMDS.out" "$input_path" -round "$round_flag" -iters "$ITERS" -batch "$BATCH" -block "$BLOCK" >"$stdout_file" 2>"$stderr_file"
      ;;
    gpucpuMDS)
      "$ROOT_DIR/gpucpuMDS.out" "$input_path" -nthreads "$THREADS" -round "$round_flag" -iters "$ITERS" -batch "$BATCH" -block "$BLOCK" >"$stdout_file" 2>"$stderr_file"
      ;;
    gpucpuMDS_v2)
      "$ROOT_DIR/gpucpuMDS_v2.out" "$input_path" -nthreads "$THREADS" -round "$round_flag" -iters "$ITERS" -batch "$BATCH" -block "$BLOCK" >"$stdout_file" 2>"$stderr_file"
      ;;
  esac

  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "OK" >"$status_file"
  else
    echo "FAIL($rc)" >"$status_file"
  fi
}

parse_row() {
  local impl="$1"
  local input_name="$2"
  local stderr_file="$RESULTS_DIR/${impl}-${input_name}.stderr.log"
  local status_file="$RESULTS_DIR/${impl}-${input_name}.status"
  local status
  status="$(cat "$status_file")"

  if [[ ! -s "$stderr_file" ]]; then
    printf "%-12s | %-14s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
      "$impl" "$input_name" "$status" "-" "-" "-" "-" "-" "-"
    return
  fi

  awk -v impl="$impl" -v input_name="$input_name" -v status="$status" '
    {
      c1="-"; c2="-"; cf="-";
      t1="-"; t2="-"; tt="-";
      note="-";
      for (i = 1; i <= NF; ++i) {
        if ($i == "Cost" && i + 3 <= NF) {
          c1=$(i+1); c2=$(i+2); cf=$(i+3);
        }
        if ($i == "Time(seconds)" && i + 3 <= NF) {
          t1=$(i+1); t2=$(i+2); tt=$(i+3);
        }
        if ($i == "VALID" || $i == "INVALID") {
          note=$i;
        }
      }
      printf "%-12s | %-14s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n",
        impl, input_name, status, c1, c2, cf, t1, t2, tt " " note;
      exit;
    }
  ' "$stderr_file"
}

echo "Results will be written to: $RESULTS_DIR"
echo "Using threads=$THREADS, iters=$ITERS, batch=$BATCH, block=$BLOCK"
echo

for input_file in "${REPRESENTATIVE_INPUTS[@]}"; do
  input_path="$INPUT_DIR/$input_file"
  if [[ ! -f "$input_path" ]]; then
    echo "Missing input: $input_path" >&2
    exit 1
  fi

  round_flag=1
  if [[ "$input_file" == Golden* ]] || [[ "$input_file" == CMT* ]]; then
    round_flag=0
  fi

  for impl in "${IMPLEMENTATIONS[@]}"; do
    echo "Running $impl on $input_file"
    run_impl "$impl" "$input_path" "$round_flag"
  done
done

echo
echo "Comparison"
printf "%-12s | %-14s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
  "Impl" "Input" "Status" "Cost1" "Cost2" "FinalCost" "Time1" "Time2" "Total"
printf "%s\n" "----------------------------------------------------------------------------------------------------------------"

for input_file in "${REPRESENTATIVE_INPUTS[@]}"; do
  input_name="$(basename "$input_file" .vrp)"
  for impl in "${IMPLEMENTATIONS[@]}"; do
    parse_row "$impl" "$input_name"
  done
done

echo
echo "All binaries are preserved in: $ROOT_DIR"
echo "All logs are preserved in: $RESULTS_DIR"
