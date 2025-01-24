#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)  # Export variables to the environment
else
    echo ".env file not found"
    exit 1
fi

# Define model names and common arguments
model_name="gemma2:27b-instruct-q8_0"

timestamp=$(date + "%Y-%m-%dT%H:%M:%S")
output_file="output_${timestamp}.log"

from_sample=0
to_sample=5000000
batch_size=100

python 01a_gpt_explanations.py \
    --model_name $model_name \
    --batch_size $batch_size \
    --from_sample $from_sample \
    --to_sample $to_sample \
    --force_rewrite \
    --use_ollama \
    --input_data_name "data/input/64_way/examples_800k_unique.jsonl" \
    --input_generated_relevance "data/extracted_relevancy.tsv" \
    --generate_into_dir "data/generated_batches/64_way" \
    | tee "output/$output_file"

