import os
import pandas as pd

# Get parquet data first
# wget https://huggingface.co/datasets/bclavie/msmarco-2m-triplets/resolve/main/data/train-00000-of-00003.parquet
# wget https://huggingface.co/datasets/bclavie/msmarco-2m-triplets/resolve/main/data/train-00001-of-00003.parquet
# wget https://huggingface.co/datasets/bclavie/msmarco-2m-triplets/resolve/main/data/train-00002-of-00003.parquet


# List of Parquet files
files = ["train-00000-of-00003.parquet", "train-00001-of-00003.parquet", "train-00002-of-00003.parquet"]
data_dir = "data/bclavie-msmarco-2m-triplets"
for file in files:
    # Read Parquet file
    df = pd.read_parquet(os.path.join(data_dir, file))

    # Save as JSON (records format for line-by-line JSON)
    json_file = file.replace(".parquet", ".jsonl")
    df.to_json(os.path.join(data_dir, json_file), orient="records", lines=True)
    print(f"Converted {file} to {json_file}")