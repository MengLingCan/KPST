#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    echo "Usage: $0 [source_domain target_domain [seed]], defaults to PC UP 25"
    echo "Environment variables: PYTHON GPU DATA_PATH PAVIA_DATA_PATH HOUSTON_DATA_PATH RESULTS_ROOT"
    exit 0
fi
if [[ $# != 0 && $# != 2 && $# != 3 ]]; then
    echo "Usage: $0 [source_domain target_domain [seed]]" >&2
    exit 2
fi

source_domain=${1:-PC}
target_domain=${2:-UP}
seed=${3:-25}
case "$source_domain:$target_domain" in
    PC:UP|UP:PC) dataset=pavia; data_path=${PAVIA_DATA_PATH:-./Pavia} ;;
    H13:H18|H18:H13) dataset=houston; data_path=${HOUSTON_DATA_PATH:-./Houston} ;;
    *) echo "Supported tasks: PC UP, UP PC, H13 H18, H18 H13" >&2; exit 2 ;;
esac
if [[ ! $seed =~ ^[0-9]{1,10}$ ]] || (( 10#$seed > 4294967295 )); then
    echo "The random seed must be an integer between 0 and 4294967295" >&2
    exit 2
fi
seed=$((10#$seed))

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
python=${PYTHON:-python}
gpu=${GPU:-0}
data_path=${DATA_PATH:-$data_path}
save_path="${RESULTS_ROOT:-./results}/$dataset/${source_domain}_to_${target_domain}/seed_$seed"
common=(--dataset "$dataset" --data_path "$data_path" --save_path "$save_path"
        --source_domain "$source_domain" --seed "$seed" --gpu "$gpu")

echo "Task: $source_domain -> $target_domain | seed=$seed | results=$save_path"
"$python" -u source_train.py "${common[@]}"
"$python" -u target_adapt.py "${common[@]}" --target_domain "$target_domain" --model_path "$save_path"
"$python" -u eval.py "${common[@]}" --target_domain "$target_domain" --eval_part both
