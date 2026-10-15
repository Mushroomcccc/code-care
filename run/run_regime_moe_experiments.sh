#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/choxia/anaconda3/envs/finrl/bin/python}"

REGIME_MOE_MARKETS="${REGIME_MOE_MARKETS:-US JAN HKEX LSE ASX}"
REGIME_MOE_SEEDS="${REGIME_MOE_SEEDS:-0 2 4 8 16}"
REGIME_MOE_WORKERS="${REGIME_MOE_WORKERS:-mimas1:0 mimas1:0 mimas1:0 mimas1:1 mimas1:1 mimas1:1 mimas2:0 mimas2:0 mimas2:0 mimas2:1 mimas2:1 mimas2:1}"
WAIT_FOR_WORKER_HOSTS="${WAIT_FOR_WORKER_HOSTS:-1}"
read -r -a MARKETS <<< "$REGIME_MOE_MARKETS"
read -r -a SEEDS <<< "$REGIME_MOE_SEEDS"
read -r -a WORKERS <<< "$REGIME_MOE_WORKERS"
SUPPORTED_PRIOR_MODELS=(llama3.1 mistral glm4 gemma4:e2b gemma4:e4b)
if [[ "$#" -gt 1 ]]; then
    echo "Usage: $0 [llama3.1|mistral|glm4|gemma4:e2b|gemma4:e4b]" >&2
    exit 2
fi
# A positional model is convenient for an interactive single-model launch;
# PRIOR_MODEL remains supported for batch orchestration and reproducibility.
PRIOR_MODEL="${1:-${PRIOR_MODEL:-gemma4:e4b}}"
ABLATION_MODE="${ABLATION_MODE:-full}"
SUPPORTED_ABLATIONS=(full no_llm_prior llm_only no_confidence single_expert)

ablation_supported=false
for supported_ablation in "${SUPPORTED_ABLATIONS[@]}"; do
    if [[ "$ABLATION_MODE" == "$supported_ablation" ]]; then
        ablation_supported=true
        break
    fi
done
if [[ "$ablation_supported" != true ]]; then
    echo "Unsupported ABLATION_MODE: $ABLATION_MODE" >&2
    echo "Supported modes: ${SUPPORTED_ABLATIONS[*]}" >&2
    exit 2
fi

prior_model_supported=false
for supported_model in "${SUPPORTED_PRIOR_MODELS[@]}"; do
    if [[ "$PRIOR_MODEL" == "$supported_model" ]]; then
        prior_model_supported=true
        break
    fi
done
if [[ "$prior_model_supported" != true ]]; then
    echo "Unsupported PRIOR_MODEL: $PRIOR_MODEL" >&2
    echo "Supported models: ${SUPPORTED_PRIOR_MODELS[*]}" >&2
    exit 2
fi

# LLM identifiers use ':' while cache filenames use a filesystem-safe '_'.
# Example: gemma4:e4b -> regime_prior_US_gemma4_e4b.csv.
PRIOR_MODEL_SLUG="${PRIOR_MODEL//:/_}"
# The Python entry point additionally normalises punctuation in result tags.
# Keep the cache slug separate because llama3.1 cache filenames retain '.'.
PRIOR_MODEL_RESULT_SLUG="${PRIOR_MODEL_SLUG//./_}"

EPOCHS="${EPOCHS:-50}"
EXPERT_PRETRAIN_EPOCHS="${EXPERT_PRETRAIN_EPOCHS:-3}"
ROUTER_PRETRAIN_EPOCHS="${ROUTER_PRETRAIN_EPOCHS:-2}"
EVAL_EVERY="${EVAL_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
ACTOR_LR="${ACTOR_LR:-0.000001}"
CRITIC_LR="${CRITIC_LR:-0.001}"
ROUTER_LR="${ROUTER_LR:-0.000001}"
ENT_COEF="${ENT_COEF:-0.0}"
MIN_CONCENTRATION="${MIN_CONCENTRATION:-0.01}"
EXPERT_LOGIT_SCALE="${EXPERT_LOGIT_SCALE:-1.0}"
ASSET_FEATURE_PROFILE="${ASSET_FEATURE_PROFILE:-method_specialization}"
ASSET_SCORER_TYPE="${ASSET_SCORER_TYPE:-linear}"
ASSET_SCORER_INIT="${ASSET_SCORER_INIT:-regime_template}"
CASH_SCORER_INIT="${CASH_SCORER_INIT:-regime_template}"
PORTFOLIO_TOP_K="${PORTFOLIO_TOP_K:-0}"
HIDDEN_SIZES="${HIDDEN_SIZES:-64 32}"
ROUTER_HIDDEN_SIZES="${ROUTER_HIDDEN_SIZES:-64 32}"
read -r -a HIDDEN_SIZE_ARGS <<< "$HIDDEN_SIZES"
read -r -a ROUTER_HIDDEN_SIZE_ARGS <<< "$ROUTER_HIDDEN_SIZES"
LAMBDA_LB="${LAMBDA_LB:-0.05}"
LAMBDA_SPEC="${LAMBDA_SPEC:-0.05}"
ROUTER_KL_COEF="${ROUTER_KL_COEF:-0.0}"
ROUTER_REWARD_COEF="${ROUTER_REWARD_COEF:-5.0}"
ROUTER_RELIABILITY_COEF="${ROUTER_RELIABILITY_COEF:-0.0}"
EXPERT_RESPONSIBILITY_ALPHA="${EXPERT_RESPONSIBILITY_ALPHA:-0.10}"
ROUTER_CAUSAL_KL_COEF="${ROUTER_CAUSAL_KL_COEF:-0.0}"
CAUSAL_PRIOR_STRENGTH="${CAUSAL_PRIOR_STRENGTH:-0.0}"
LLM_PRIOR_STRENGTH="${LLM_PRIOR_STRENGTH:-1.0}"
JOINT_KL_FINAL_MULT="${JOINT_KL_FINAL_MULT:-0.1}"
DIRECT_RETURN_COEF="${DIRECT_RETURN_COEF:-1.0}"
REFIT_EPOCHS="${REFIT_EPOCHS:-0}"
DIRECT_DOWNSIDE_COEF="${DIRECT_DOWNSIDE_COEF:-0.0}"
ANCHOR_STRENGTH="${ANCHOR_STRENGTH:-0.0}"
ANCHOR_STYLE_STRENGTH="${ANCHOR_STYLE_STRENGTH:-0.0}"
CHECKPOINT_SELECTION_METRIC="${CHECKPOINT_SELECTION_METRIC:-cumret}"
VALIDATION_ONLY="${VALIDATION_ONLY:-0}"
VALIDATION_REFIT_PROBE="${VALIDATION_REFIT_PROBE:-0}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
RESULTS_BASE="${RESULTS_BASE:-results_regime_moe_formal}"
RUN_TAG="${RUN_TAG:-method_trainable_templates_confinteraction_v4p1_posttest_style_a1e6_rr5_rlr1e6_s1_dr1_refit0_eval5_${PRIOR_MODEL_RESULT_SLUG}}"
RESULTS_ROOT="${PROJECT_ROOT}/${RESULTS_BASE}_epo${EPOCHS}_exp${EXPERT_PRETRAIN_EPOCHS}_rt${ROUTER_PRETRAIN_EPOCHS}_bs${BATCH_SIZE}_lb${LAMBDA_LB}_sp${LAMBDA_SPEC}_kl${ROUTER_KL_COEF}_${RUN_TAG}"
DRIVER_LOG_DIR="${PROJECT_ROOT}/run/logs/regime_moe_${RUN_TAG}"
mkdir -p "$DRIVER_LOG_DIR"

# Do not consume the task queue when a requested training host is temporarily
# unreachable. This is especially important when switching server pools
# between model waves: a failed SSH fan-out must not be mistaken for a failed
# experiment grid or silently fall back to another host.
if [[ "$WAIT_FOR_WORKER_HOSTS" == "1" ]]; then
    worker_hosts=()
    for worker in "${WORKERS[@]}"; do
        host="${worker%%:*}"
        if [[ ! " ${worker_hosts[*]} " =~ " ${host} " ]]; then
            worker_hosts+=("$host")
        fi
    done
    while true; do
        unavailable_hosts=()
        for host in "${worker_hosts[@]}"; do
            if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" true; then
                unavailable_hosts+=("$host")
            fi
        done
        [[ "${#unavailable_hosts[@]}" -eq 0 ]] && break
        echo "Waiting for requested worker hosts: ${unavailable_hosts[*]}"
        sleep 30
    done
fi

for market in "${MARKETS[@]}"; do
    for required in \
        "${PROJECT_ROOT}/data/${market}_54.npz" \
        "${PROJECT_ROOT}/data/${market}_snames9.npz" \
        "${PROJECT_ROOT}/run/cache/regime_prior_${market}_${PRIOR_MODEL_SLUG}.csv"; do
        if [[ ! -f "$required" ]]; then
            echo "Missing required input: $required" >&2
            exit 1
        fi
    done
done

tasks=()
for market in "${MARKETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tasks+=("${market}:${seed}")
    done
done

queue_file="$(mktemp)"
failure_file="$(mktemp)"
lock_file="${queue_file}.lock"
touch "$lock_file"
trap 'rm -f "$queue_file" "$failure_file" "$lock_file"' EXIT
printf '%s\n' "${tasks[@]}" > "$queue_file"

take_task() {
    local task
    exec 200>"$lock_file"
    flock 200
    task="$(head -n 1 "$queue_file")"
    if [[ -n "$task" ]]; then
        tail -n +2 "$queue_file" > "${queue_file}.tmp"
        mv "${queue_file}.tmp" "$queue_file"
    fi
    flock -u 200
    exec 200>&-
    printf '%s' "$task"
}

run_worker() {
    local worker="$1"
    local host="${worker%%:*}"
    local gpu="${worker##*:}"
    local task market seed prior log_path complete_count command_text
    while true; do
        task="$(take_task)"
        [[ -z "$task" ]] && break
        market="${task%%:*}"
        seed="${task##*:}"
        prior="run/cache/regime_prior_${market}_${PRIOR_MODEL_SLUG}.csv"
        log_path="${DRIVER_LOG_DIR}/${market}_seed${seed}_${host}_gpu${gpu}.log"

        complete_count=0
        if [[ -d "${RESULTS_ROOT}/${market}" ]]; then
            complete_count="$($PYTHON_BIN - "${RESULTS_ROOT}/${market}" "$seed" <<'PY'
import json, pathlib, sys
root, seed = pathlib.Path(sys.argv[1]), int(sys.argv[2])
count = 0
for path in root.glob(f"manifest_{seed}_*.json"):
    try:
        count += json.loads(path.read_text()).get("status") == "complete"
    except Exception:
        pass
print(count)
PY
)"
        fi
        if [[ "$complete_count" -gt 0 ]]; then
            echo "SKIP complete market=${market} seed=${seed}"
            continue
        fi

        printf -v command_text '%q ' \
            "$PYTHON_BIN" run/run_regime_moe.py \
            --market "$market" \
            --num-sectors 9 \
            --device cuda:0 \
            --seed "$seed" \
            --epochs "$EPOCHS" \
            --refit-epochs "$REFIT_EPOCHS" \
            --expert-pretrain-epochs "$EXPERT_PRETRAIN_EPOCHS" \
            --router-pretrain-epochs "$ROUTER_PRETRAIN_EPOCHS" \
            --eval-every "$EVAL_EVERY" \
            --batch-size "$BATCH_SIZE" \
            --ppo-epochs "$PPO_EPOCHS" \
            --actor-lr "$ACTOR_LR" \
            --critic-lr "$CRITIC_LR" \
            --router-lr "$ROUTER_LR" \
            --hidden-sizes "${HIDDEN_SIZE_ARGS[@]}" \
            --router-hidden-sizes "${ROUTER_HIDDEN_SIZE_ARGS[@]}" \
            --ent-coef "$ENT_COEF" \
            --min-concentration "$MIN_CONCENTRATION" \
            --expert-logit-scale "$EXPERT_LOGIT_SCALE" \
            --asset-feature-profile "$ASSET_FEATURE_PROFILE" \
            --asset-scorer-type "$ASSET_SCORER_TYPE" \
            --asset-scorer-init "$ASSET_SCORER_INIT" \
            --cash-scorer-init "$CASH_SCORER_INIT" \
            --portfolio-top-k "$PORTFOLIO_TOP_K" \
            --lambda-lb "$LAMBDA_LB" \
            --lambda-spec "$LAMBDA_SPEC" \
            --router-kl-coef "$ROUTER_KL_COEF" \
            --router-reward-coef "$ROUTER_REWARD_COEF" \
            --router-reliability-coef "$ROUTER_RELIABILITY_COEF" \
            --expert-responsibility-alpha "$EXPERT_RESPONSIBILITY_ALPHA" \
            --router-causal-kl-coef "$ROUTER_CAUSAL_KL_COEF" \
            --causal-prior-strength "$CAUSAL_PRIOR_STRENGTH" \
            --llm-prior-strength "$LLM_PRIOR_STRENGTH" \
            --joint-kl-final-mult "$JOINT_KL_FINAL_MULT" \
            --direct-return-coef "$DIRECT_RETURN_COEF" \
            --direct-downside-coef "$DIRECT_DOWNSIDE_COEF" \
            --anchor-strength "$ANCHOR_STRENGTH" \
            --anchor-style-strength "$ANCHOR_STYLE_STRENGTH" \
            --ablation-mode "$ABLATION_MODE" \
            --checkpoint-selection-metric "$CHECKPOINT_SELECTION_METRIC" \
            --prior-mode cache \
            --prior-cache "$prior" \
            --no-random-reset \
            --deterministic \
            --ckpt auto \
            --results_dir_name "$RESULTS_BASE" \
            --run-tag "$RUN_TAG"

        if [[ "$VALIDATION_REFIT_PROBE" == "1" ]]; then
            command_text+=" --validation-refit-probe"
        elif [[ "$VALIDATION_ONLY" == "1" ]]; then
            command_text+=" --validation-only"
        fi

        echo "START worker=${worker} market=${market} seed=${seed}"
        if ssh -o BatchMode=yes "$host" \
            "cd $(printf '%q' "$PROJECT_ROOT") && CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu") $command_text" \
            > "$log_path" 2>&1; then
            echo "DONE  worker=${worker} market=${market} seed=${seed}"
        else
            echo "FAIL  worker=${worker} market=${market} seed=${seed}; log=${log_path}" >&2
            printf '%s\n' "${market}:${seed}:${worker}:${log_path}" >> "$failure_file"
        fi
    done
}

echo "Fixed grid: markets=${MARKETS[*]} seeds=${SEEDS[*]}"
echo "Workers: ${WORKERS[*]}"
echo "LLM prior model: ${PRIOR_MODEL} (cache slug: ${PRIOR_MODEL_SLUG}; result slug: ${PRIOR_MODEL_RESULT_SLUG})"
echo "Ablation mode: ${ABLATION_MODE}"
echo "Training epochs: ${EPOCHS}"
echo "Results: ${RESULTS_ROOT}"
worker_pids=()
for worker in "${WORKERS[@]}"; do
    run_worker "$worker" &
    worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
    wait "$pid" || true
done

if [[ -s "$failure_file" ]]; then
    echo "Failed tasks:" >&2
    sed -n '1,200p' "$failure_file" >&2
    exit 1
fi

if [[ "$VALIDATION_ONLY" == "1" || "$VALIDATION_REFIT_PROBE" == "1" || "$SKIP_ANALYSIS" == "1" || "${#MARKETS[@]}" -ne 5 || "${#SEEDS[@]}" -ne 5 ]]; then
    echo "Training grid completed; aggregate formal analysis skipped for validation/partial grid."
    exit 0
fi

analysis_args=(
    "$PYTHON_BIN"
    "${PROJECT_ROOT}/analyse/analyse_regime_moe.py"
    --results-root "$RESULTS_ROOT"
)
if [[ "$PRIOR_MODEL" != "gemma4:e4b" || "$ABLATION_MODE" != "full" ]]; then
    analysis_args+=(--overall-only)
fi
if [[ "$ABLATION_MODE" == "full" ]]; then
    analysis_args+=(--require-overall-pass)
fi
"${analysis_args[@]}"
echo "All fixed-grid experiments and analysis completed."
