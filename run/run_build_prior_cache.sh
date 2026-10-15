#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "$REPO_DIR"

DATA_DIR="${PROJECT_DIR}/data"
STOCKS_PER_SECTOR="${STOCKS_PER_SECTOR:-6}"
NUM_SECTORS="${NUM_SECTORS:-9}"
NUM_STOCKS=$((NUM_SECTORS * STOCKS_PER_SECTOR))

if [ -n "${MARKETS:-}" ]; then
  read -r -a MARKET_LIST <<< "$MARKETS"
else
  mapfile -t MARKET_LIST < <(
    find "$DATA_DIR" -maxdepth 1 -type f -name "*_${NUM_STOCKS}.npz" -printf '%f\n' \
      | sed "s/_${NUM_STOCKS}\\.npz$//" \
      | sort
  )
fi

if [ -n "${PORTS:-}" ]; then
  read -r -a PORT_LIST <<< "$PORTS"
else
  PORT_LIST=(6019 6020)
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_MODELS="gemma4:e4b"
# llama3.1 mistral glm4 gemma4:e2b gemma4:e4b"
if [ -n "${MODELS:-}" ]; then
  read -r -a MODEL_LIST <<< "$MODELS"
elif [ -n "${MODEL:-}" ]; then
  read -r -a MODEL_LIST <<< "$MODEL"
else
  read -r -a MODEL_LIST <<< "$DEFAULT_MODELS"
fi
MODE="${MODE:-llm}"
CONTEXT_MODEL="${CONTEXT_MODEL:-}"
SEED="${SEED:-2}"
GAMMA="${GAMMA:-0.99}"
RISK_WINDOW="${RISK_WINDOW:-20}"
SPLIT="${SPLIT:-full}"
LIMIT="${LIMIT:-0}"
START_DAY="${START_DAY:-}"
END_DAY="${END_DAY:-}"
SAMPLES="${SAMPLES:-5}"
LLM_TIMEOUT="${LLM_TIMEOUT:-40}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-3}"
SUMMARY_TEMPERATURE="${SUMMARY_TEMPERATURE:-0.0}"
REGIME_TEMPERATURE="${REGIME_TEMPERATURE:-0.7}"
NEWS_LOOKBACK_DAYS="${NEWS_LOOKBACK_DAYS:-1}"
MAX_SECTOR_TITLES="${MAX_SECTOR_TITLES:-8}"
MAX_STOCK_TITLES="${MAX_STOCK_TITLES:-3}"
MAX_STOCK_ITEMS="${MAX_STOCK_ITEMS:-30}"
SAVE_EVERY="${SAVE_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
OVERWRITE="${OVERWRITE:-0}"
NO_JSON_CACHE="${NO_JSON_CACHE:-0}"
MAX_ATTEMPTS_PER_TASK="${MAX_ATTEMPTS_PER_TASK:-0}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/cache}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/build_prior_cache}"

if [ "${#MARKET_LIST[@]}" -eq 0 ]; then
  echo "No ${NUM_STOCKS}-stock datasets found in ${DATA_DIR}."
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN"
  exit 1
fi

for market in "${MARKET_LIST[@]}"; do
  if [[ ! "$market" =~ ^[[:alnum:]_-]+$ ]]; then
    echo "Invalid market name: $market"
    exit 1
  fi
  for required in \
    "$DATA_DIR/${market}_${NUM_STOCKS}.npz" \
    "$DATA_DIR/${market}_snames${NUM_SECTORS}.npz" \
    "$DATA_DIR/${market}_stock_with_news.csv"; do
    if [ ! -f "$required" ]; then
      echo "Missing dataset file: $required"
      exit 1
    fi
  done
done

if [ "${#PORT_LIST[@]}" -eq 0 ]; then
  echo "No LLM ports configured. Set PORTS='6019 6020'."
  exit 1
fi

if [ "${#MODEL_LIST[@]}" -eq 0 ]; then
  echo "No models configured. Set MODELS='llama3.1 mistral glm4 gemma4:e4b'."
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

model_slug() {
  local slug
  slug="$(printf '%s' "$1" | tr -c '[:alnum:]_.-' '_' | sed 's/^_*//; s/_*$//')"
  if [ -z "$slug" ]; then
    slug="model"
  fi
  printf '%s' "$slug"
}

check_cache_complete() {
  local market="$1"
  local out_path="$2"
  "$PYTHON_BIN" - "$PROJECT_DIR" "$market" "$out_path" "$NUM_SECTORS" "$GAMMA" "$RISK_WINDOW" "$SPLIT" "$LIMIT" "$START_DAY" "$END_DAY" "$CONTEXT_MODEL" <<'PY'
import argparse
import csv
import os
import sys

project_dir = sys.argv[1]
market = sys.argv[2]
out_path = sys.argv[3]
num_sectors = int(sys.argv[4])
gamma = float(sys.argv[5])
risk_window = int(sys.argv[6])
split = sys.argv[7]
limit = int(sys.argv[8])
start_day = int(sys.argv[9]) if sys.argv[9] else None
end_day = int(sys.argv[10]) if sys.argv[10] else None
context_model = sys.argv[11] or None

sys.path.insert(0, project_dir)

from run.build_regime_prior_cache import _day_bounds
from run.common.envs import make_stock_env

env = make_stock_env(
    0,
    10**9,
    market,
    sector_num=num_sectors,
    project_root=project_dir,
    env_name="prior_full_check",
    gamma=gamma,
    risk_window=risk_window,
    vol_penalty_coef=0.0,
    drawdown_penalty_coef=0.0,
    context_model=context_model,
)
env.if_random_reset = False
env.reset()
cfg = argparse.Namespace(split=split, start_day=start_day, end_day=end_day, limit=limit)
start, end = _day_bounds(int(env.max_step), cfg)
expected = set(range(start, end + 1))

actual = set()
if os.path.exists(out_path):
    with open(out_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                actual.add(int(float(str(row.get("absolute_day", "")).strip())))
            except (TypeError, ValueError):
                pass

missing = sorted(expected - actual)
extra = sorted(actual - expected)
prefix = f"expected={len(expected)} actual={len(actual)} missing={len(missing)} extra={len(extra)} range=[{start},{end}]"
if missing:
    print(f"{prefix} first_missing={missing[0]} last_missing={missing[-1]}")
    sys.exit(1)
print(prefix)
PY
}

TASK_MARKETS=()
TASK_MODELS=()
for model in "${MODEL_LIST[@]}"; do
  for market in "${MARKET_LIST[@]}"; do
    TASK_MARKETS+=("$market")
    TASK_MODELS+=("$model")
  done
done

echo "Building Regime-MoE LLM prior caches"
echo "  markets: ${MARKET_LIST[*]}"
echo "  models:  ${MODEL_LIST[*]}"
echo "  ports:   ${PORT_LIST[*]}"
echo "  mode:    ${MODE}"
echo "  split:   ${SPLIT}"
echo "  out:     ${OUT_DIR}"
echo "  logs:    ${LOG_DIR}"
echo "  retry:   max_attempts_per_task=${MAX_ATTEMPTS_PER_TASK} max_rounds=${MAX_ROUNDS} sleep=${RETRY_SLEEP}s"

run_task_once() {
  local task_idx="$1"
  local port="$2"
  local attempt="$3"
  local overwrite_this_attempt="$4"
  local market="${TASK_MARKETS[$task_idx]}"
  local model="${TASK_MODELS[$task_idx]}"
  local slug
  slug="$(model_slug "$model")"
  local out_path="${OUT_DIR}/regime_prior_${market}_${slug}.csv"
  local json_cache_path="${OUT_DIR}/regime_prior_llm_cache_${market}_${slug}.json"
  local log_path="${LOG_DIR}/${market}_${slug}_port${port}.log"
  local exit_code=0
  local status_msg
  local cmd=(
    "$PYTHON_BIN" "$SCRIPT_DIR/build_regime_prior_cache.py"
    --market "$market"
    --num-sectors "$NUM_SECTORS"
    --seed "$SEED"
    --gamma "$GAMMA"
    --risk-window "$RISK_WINDOW"
    --model "$model"
    --port "$port"
    --mode "$MODE"
    --llm-timeout "$LLM_TIMEOUT"
    --llm-max-retries "$LLM_MAX_RETRIES"
    --samples "$SAMPLES"
    --summary-temperature "$SUMMARY_TEMPERATURE"
    --regime-temperature "$REGIME_TEMPERATURE"
    --news-lookback-days "$NEWS_LOOKBACK_DAYS"
    --max-sector-titles "$MAX_SECTOR_TITLES"
    --max-stock-titles "$MAX_STOCK_TITLES"
    --max-stock-items "$MAX_STOCK_ITEMS"
    --split "$SPLIT"
    --limit "$LIMIT"
    --out "$out_path"
    --cache-path "$json_cache_path"
    --save-every "$SAVE_EVERY"
    --log-every "$LOG_EVERY"
  )

  if [ -n "$CONTEXT_MODEL" ]; then
    cmd+=(--context-model "$CONTEXT_MODEL")
  fi
  if [ -n "$START_DAY" ]; then
    cmd+=(--start-day "$START_DAY")
  fi
  if [ -n "$END_DAY" ]; then
    cmd+=(--end-day "$END_DAY")
  fi
  if [ "$overwrite_this_attempt" = "1" ]; then
    cmd+=(--overwrite)
  fi
  if [ "$NO_JSON_CACHE" = "1" ]; then
    cmd+=(--no-json-cache)
  fi

  if [ "$attempt" -eq 1 ]; then
    : > "$log_path"
  fi

  {
    echo
    echo ">>> [port ${port}] attempt=${attempt} model=${model} market=${market} -> ${out_path}"
    if status_msg="$(check_cache_complete "$market" "$out_path")"; then
      echo "    pre-check: complete ${status_msg}"
    else
      echo "    pre-check: incomplete ${status_msg}"
    fi
  } | tee -a "$log_path"

  if "${cmd[@]}" 2>&1 | tee -a "$log_path"; then
    exit_code=0
  else
    exit_code=$?
  fi

  if [ "$exit_code" -ne 0 ]; then
    echo "    command exit_code=${exit_code}" | tee -a "$log_path"
  fi
  return "$exit_code"
}

run_worker() {
  local worker_id="$1"
  local port="$2"
  local round=0
  local pending=0
  local ran_any=0
  local maxed=0
  local idx
  local attempts
  local overwrite_this_attempt
  local market
  local model
  local slug
  local out_path
  local status_msg
  local task_attempts=()

  while true; do
    round=$((round + 1))
    pending=0
    ran_any=0
    maxed=0
    echo "=== worker=${worker_id} port=${port} round=${round} ==="

    for idx in "${!TASK_MARKETS[@]}"; do
      if (( idx % ${#PORT_LIST[@]} != worker_id )); then
        continue
      fi

      market="${TASK_MARKETS[$idx]}"
      model="${TASK_MODELS[$idx]}"
      slug="$(model_slug "$model")"
      out_path="${OUT_DIR}/regime_prior_${market}_${slug}.csv"
      attempts="${task_attempts[$idx]:-0}"

      if ! { [ "$OVERWRITE" = "1" ] && [ "$attempts" -eq 0 ]; }; then
        if status_msg="$(check_cache_complete "$market" "$out_path")"; then
          echo "OK: model=${model} market=${market} ${status_msg}"
          continue
        fi
      else
        status_msg="overwrite requested for first attempt"
      fi

      pending=$((pending + 1))
      echo "PENDING: model=${model} market=${market} ${status_msg}"

      if [ "$MAX_ATTEMPTS_PER_TASK" -gt 0 ] && [ "$attempts" -ge "$MAX_ATTEMPTS_PER_TASK" ]; then
        echo "MAX_ATTEMPTS_REACHED: model=${model} market=${market} attempts=${attempts}"
        maxed=1
        continue
      fi

      attempts=$((attempts + 1))
      task_attempts[$idx]="$attempts"
      overwrite_this_attempt=0
      if [ "$OVERWRITE" = "1" ] && [ "$attempts" -eq 1 ]; then
        overwrite_this_attempt=1
      fi

      if ! run_task_once "$idx" "$port" "$attempts" "$overwrite_this_attempt"; then
        echo "FAILED_ATTEMPT: model=${model} market=${market} port=${port} attempt=${attempts}"
      fi
      if status_msg="$(check_cache_complete "$market" "$out_path")"; then
        echo "COMPLETED: model=${model} market=${market} ${status_msg}"
        pending=$((pending - 1))
      else
        echo "STILL_INCOMPLETE: model=${model} market=${market} ${status_msg}"
      fi
      ran_any=1
    done

    if [ "$pending" -eq 0 ]; then
      echo "worker=${worker_id} port=${port}: all assigned caches complete."
      return 0
    fi
    if [ "$maxed" -ne 0 ] && [ "$ran_any" -eq 0 ]; then
      echo "worker=${worker_id} port=${port}: unfinished tasks reached MAX_ATTEMPTS_PER_TASK."
      return 1
    fi
    if [ "$MAX_ROUNDS" -gt 0 ] && [ "$round" -ge "$MAX_ROUNDS" ]; then
      echo "worker=${worker_id} port=${port}: unfinished tasks remain after MAX_ROUNDS=${MAX_ROUNDS}."
      return 1
    fi

    echo "worker=${worker_id} port=${port}: ${pending} task(s) still incomplete; retrying after ${RETRY_SLEEP}s."
    sleep "$RETRY_SLEEP"
  done
}

pids=()
for idx in "${!PORT_LIST[@]}"; do
  run_worker "$idx" "${PORT_LIST[$idx]}" &
  pids+=("$!")
done

fail_count=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail_count=$((fail_count + 1))
  fi
done

if [ "$fail_count" -ne 0 ]; then
  echo "Finished with ${fail_count} failed worker(s). See logs in ${LOG_DIR}."
  exit 1
fi

echo "All Regime-MoE prior caches built successfully."
