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
  "llama3.1:8b-instruct-fp16 --use_ollama"
  "llama3.1:70b-instruct-q4_0 --use_ollama"
  "gemma2:9b-instruct-q8_0 --use_ollama"
  "gemma2:27b-instruct-q8_0 --use_ollama"
  "gpt-4o-2024-08-06"
  "gpt-4o-mini-2024-07-18"
  "llama3.1:70b-instruct-q8_0 --use_ollama"
)

# Iterate over the models
for model_args in "${models[@]}"; do
  python 01a_gpt_explanations.py --model_name $model_args --batch_size 15 --to_sample 35 --from_sample 30 --force_rewrite
  echo -e "\n\n"  
done

