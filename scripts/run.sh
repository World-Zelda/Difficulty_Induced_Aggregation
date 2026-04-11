#!/bin/bash
models=(
  'Qwen/Qwen3-0.6B'
)

datasets=(
  "HuggingFaceH4/MATH-500"
)

temperatures=(0.4 0.6 0.8 1.0 1.2)

NUM_SAMPLES=10
OUTPUT_BASE_DIR="results"

mkdir -p "$OUTPUT_BASE_DIR"

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for temp in "${temperatures[@]}"; do
      echo "Running: model=${model}, dataset=${dataset}, temperature=${temp}"

      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "$PYTHON" inference.py \
        --model_name "$model" \
        --dataset "$dataset" \
        --do_sample \
        --num_samples "$NUM_SAMPLES" \
        --temperature "$temp" \
        --output_base_dir "$OUTPUT_BASE_DIR"

      echo "Finished: ${model##*/} | ${dataset##*/} | T=${temp}"
      echo "----------------------------------------"
    done
  done
done

echo "All experiments completed!"