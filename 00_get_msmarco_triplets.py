import jsonlines
import argparse
import os
import subprocess
import pandas as pd
from tqdm import tqdm


def convert_from_parquet(data_dir):
    files = ["train-00000-of-00003.parquet", "train-00001-of-00003.parquet", "train-00002-of-00003.parquet"]
    for file in files:
        # Read Parquet file
        df = pd.read_parquet(os.path.join(data_dir, file))
    
        # Save as JSON (records format for line-by-line JSON)
        json_file = file.replace(".parquet", ".jsonl")
        df.to_json(os.path.join(data_dir, json_file), orient="records", lines=True)
        print(f"Converted {file} to {json_file}")


def download_data():
    urls = [
        "https://huggingface.co/datasets/bclavie/msmarco-2m-triplets/resolve/main/data/train-00000-of-00003.parquet",
        "https://huggingface.co/datasets/bclavie/msmarco-2m-triplets/resolve/main/data/train-00001-of-00003.parquet",
        "https://huggingface.co/datasets/bclavie/msmarco-2m-triplets/resolve/main/data/train-00002-of-00003.parquet",
    ]

    for url in urls:
        subprocess.run(["wget", url], check=True)


def get_args():
    parser = argparse.ArgumentParser("Create triplets in (tsv) format compatible for ColBERT training")

    parser.add_argument("--nr_triplets", type=int, default=10_000,
                        help="Number of triplets to create")
    parser.add_argument("--data_dir", type=str, default="data/bclavie-msmarco-2m-triplets",
                        help="Data dir to download data and construct triplets.")
    parser.add_argument("--input_file_name", type=str, default="train-00000-of-00003.jsonl")
    return parser.parse_args()


def main():
    args = get_args()

    data_dir = args.data_dir
    nr_triplets = args.nr_triplets
    input_file_name = os.path.join(data_dir, args.input_file_name)

    if not os.path.exists(data_dir):
        download_data()
        convert_from_parquet(data_dir)

    input_data = []
    with jsonlines.open(input_file_name) as reader:
        for line_obj in reader:
            input_data.append(line_obj)

    triplets = []
    collection = dict()
    queries = dict()
    for triplet in tqdm(input_data[:nr_triplets]):
        query = triplet['query']
        positive = triplet['positive']
        negative = triplet['negative']

        # Add passages to collection if not present already
        for psg in [positive, negative]:
            if psg not in collection.keys():
                collection[psg] = len(collection)

        # Add query to collection if not present already
        if query not in queries.keys():
            queries[query] = len(queries)

        # Construct triplets out of ids in dictionaries
        triplets.append(
            [queries[query], collection[positive], collection[negative]]
        )

    out_dir = str(os.path.join(data_dir, str(nr_triplets)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Write out collection
    collection_out = os.path.join(out_dir, f"collection.tsv")
    with open(collection_out, mode="w") as out_file:
        for doc, doc_id in collection.items():
            out_file.write(f"{doc_id}\t{doc}\n")

    # Write out all queries
    queries_out = os.path.join(out_dir, f"queries.tsv")
    with open(queries_out, mode="w") as out_file:
        for q, q_id in queries.items():
            out_file.write(f"{q_id}\t{q}\n")

    # Write out all triplets
    triplets_out = os.path.join(out_dir, f"triplets.tsv")
    with jsonlines.open(triplets_out, mode="w") as out_file:
        out_file.write_all(triplets)


if __name__ == "__main__":
    main()