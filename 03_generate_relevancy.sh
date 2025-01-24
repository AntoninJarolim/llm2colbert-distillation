#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)  # Export variables to the environment
else
    echo ".env file not found"
    exit 1
fi

# Define model names and common arguments
model="gemma2:27b-instruct-q8_0"

python 01a_gpt_explanations.py \
    --model_name $model \
    --batch_size 100 \
    --from_sample 0 \
    --to_sample 50000 \
    --force_rewrite \
    --use_ollama \
    --input_data_name "data/input/64_way/examples_800k_unique.jsonl" \
    --input_generated_relevancy "data/extracted_relevancy.tsv" \
    --output_data_name "${model_name}.jsonl"



