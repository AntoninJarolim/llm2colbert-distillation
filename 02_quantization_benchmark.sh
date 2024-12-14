#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)  # Export variables to the environment
else
    echo ".env file not found"
    exit 1
fi

# Define model names and common arguments
models=(
 "gemma2:27b-instruct-q8_0"
 "gemma2:27b-instruct-fp16"
)

# Iterate over the models
for model_name in "${models[@]}"; do
  echo "Running benchmark for model: $model_name"
  start_time=$(date +%s)  # Record the start time

  python 01a_gpt_explanations.py \
    --model_name $model_name \
    --batch_size 15 \
    --to_sample 500 \
    --force_rewrite \
    --use_ollama \
    --input_data_name "data/bclavie-msmarco-2m-triplets/train-00000-of-00003.jsonl" \
    --output_data_name "${model_name}.jsonl"

  end_time=$(date +%s)  # Record the end time
  runtime=$((end_time - start_time))  # Calculate runtime
  echo "Runtime for model ($model_name): ${runtime}s"
  echo -e "\n\n"
done


