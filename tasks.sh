#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    echo "Usage: $0, runs all four directions in sequence with four random seeds per direction"
    echo "Environment variables: SEEDS (default '25 50 75 100') PYTHON GPU PAVIA_DATA_PATH HOUSTON_DATA_PATH RESULTS_ROOT"
    exit 0
fi
if [[ $# != 0 ]]; then
    echo "Usage: $0; use the SEEDS environment variable to change the four random seeds" >&2
    exit 2
fi

read -r -a seeds <<< "${SEEDS:-25 50 75 100}"
if [[ ${#seeds[@]} != 4 ]]; then
    echo "SEEDS must contain four distinct random seeds" >&2
    exit 2
fi
for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    if [[ ! $seed =~ ^[0-9]{1,10}$ ]] || (( 10#$seed > 4294967295 )); then
        echo "The random seed must be an integer between 0 and 4294967295" >&2
        exit 2
    fi
    seed=$((10#$seed))
    seeds[$i]=$seed
    for ((j=0; j<i; j++)); do
        if (( seed == seeds[j] )); then
            echo "SEEDS must contain four distinct random seeds" >&2
            exit 2
        fi
    done
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
for task in "PC UP" "UP PC" "H13 H18" "H18 H13"; do
    read -r source_domain target_domain <<< "$task"
    if [[ $source_domain == PC || $source_domain == UP ]]; then
        data_path=${PAVIA_DATA_PATH:-./Pavia}
    else
        data_path=${HOUSTON_DATA_PATH:-./Houston}
    fi
    for seed in "${seeds[@]}"; do
        DATA_PATH="$data_path" bash "$script_dir/auto.sh" "$source_domain" "$target_domain" "$seed"
    done
done
