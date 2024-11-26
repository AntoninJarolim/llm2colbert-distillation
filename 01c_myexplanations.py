import random
from datasets import load_dataset, Dataset
import json
from tqdm import tqdm

def print_30_random():
    # Load the dataset
    dataset = load_dataset("bclavie/msmarco-2m-triplets")["train"]

    # Slice the first 10,000 rows
    subset = dataset.select(range(10_000))

    # Randomly select 1000 rows
    select_n = 1000
    random.seed(42)  # For reproducibility 29 samples
    random.seed(56415)  # For reproducibility 6 additional samples
    sample_indices = random.sample(range(len(subset)), select_n)
    sampled_data = subset.select(sample_indices)

    # To view the sampled data
    nr_to_save = 6
    for i in range(select_n):
        if sampled_data[i]["positive"].isascii():
            print(json.dumps(sampled_data[i], ensure_ascii=False))
            # print(sampled_data[i]["positive"])

            nr_to_save -= 1
            if nr_to_save == 0:
                break


def annotate_data():
    # Path to your JSON file
    input_file = "data/6_random_samples.jsonl"
    output_file = "data/6_random_samples_explained.jsonl"

    # Open the output file for writing
    with open(output_file, "w") as outfile:
        # Open the JSONL file line by line
        with open(input_file, "r") as infile:
            for line in infile:
                # Load each line as a JSON object
                d = json.loads(line.strip())

                # Print the query and positive fields
                print(f"Query: {d['query']}")
                print(f"Text : {d['positive']}")

                # Wait for user input
                selected_spans = []
                while True:
                    selected_span = input("Enter selected span: ")
                    if selected_span == "end":
                        break
                    selected_spans.append(selected_span)

                # Write the modified JSON object back to the output file
                print("\n".join(selected_spans))
                d["selected_spans"] = selected_spans
                outfile.write(json.dumps(d, ensure_ascii=False) + "\n")

                print("\n---\n")

# print_30_random()
annotate_data()

# another empty
#  what does the ignition distributor do