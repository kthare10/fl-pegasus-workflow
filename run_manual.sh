#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment.yaml}"
NUM_CLIENTS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_clients"])' "$CONFIG")"
ROUNDS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["rounds"])' "$CONFIG")"
DATASET="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_name"])' "$CONFIG")"
if [[ "$DATASET" == tcia* ]]; then
  PREPROCESS="bin/preprocess_tcia.py"
else
  PREPROCESS="bin/preprocess_nih_cxr14.py"
fi

rm -rf manifests preprocessed models metrics results

python3 bin/prepare_manifest.py \
  --config "$CONFIG" \
  --dataset-manifest manifests/dataset_manifest.csv \
  --client-manifest manifests/client_manifest.csv \
  --splits manifests/splits.json

for ((client_id = 0; client_id < NUM_CLIENTS; client_id++)); do
  python3 "$PREPROCESS" \
    --config "$CONFIG" \
    --client-manifest manifests/client_manifest.csv \
    --client-id "$client_id" \
    --output "preprocessed/client_$(printf '%03d' "$client_id").jsonl"
done

python3 bin/initialize_model.py \
  --config "$CONFIG" \
  --model models/round_000_global.pt \
  --model-config models/model_config.json

current_model="models/round_000_global.pt"
for ((round_id = 1; round_id <= ROUNDS; round_id++)); do
  update_args=()
  count_args=()
  for ((client_id = 0; client_id < NUM_CLIENTS; client_id++)); do
    client="$(printf '%03d' "$client_id")"
    round="$(printf '%03d' "$round_id")"
    python3 bin/train_client.py \
      --config "$CONFIG" \
      --client-id "$client_id" \
      --round "$round_id" \
      --global-model "$current_model" \
      --client-data "preprocessed/client_${client}.jsonl" \
      --output-model "models/round_${round}/client_${client}_weights.pt" \
      --metrics "metrics/round_${round}/client_${client}_train.json" \
      --count-output "metrics/round_${round}/client_${client}_count.json"
    update_args+=(--client-update "models/round_${round}/client_${client}_weights.pt")
    count_args+=(--client-count "metrics/round_${round}/client_${client}_count.json")
  done
  python3 bin/aggregate.py \
    --config "$CONFIG" \
    --round "$round_id" \
    "${update_args[@]}" \
    "${count_args[@]}" \
    --output-model "models/round_${round}_global.pt" \
    --metrics "metrics/round_${round}/round_${round}_aggregation.json"
  current_model="models/round_${round}_global.pt"
done

client_data_args=()
for ((client_id = 0; client_id < NUM_CLIENTS; client_id++)); do
  client="$(printf '%03d' "$client_id")"
  client_data_args+=(--client-data "preprocessed/client_${client}.jsonl")
done

python3 bin/evaluate.py \
  --config "$CONFIG" \
  --model "$current_model" \
  "${client_data_args[@]}" \
  --output metrics/final_evaluation.json

round_metric_args=()
for ((round_id = 1; round_id <= ROUNDS; round_id++)); do
  round="$(printf '%03d' "$round_id")"
  round_metric_args+=(--round-metric "metrics/round_${round}/round_${round}_aggregation.json")
done

python3 bin/package_results.py \
  --config "$CONFIG" \
  --evaluation metrics/final_evaluation.json \
  "${round_metric_args[@]}" \
  --results results/results.tar.gz \
  --paper-tables results/paper_tables.csv \
  --provenance results/provenance_summary.json

echo "Manual smoke test complete: results/paper_tables.csv"
