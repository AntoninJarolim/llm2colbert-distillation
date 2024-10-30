import json

import transformers
import torch
from datasets import load_dataset
from huggingface_hub import login
import os

from tqdm import tqdm

login(token=os.getenv("HUGGINGFACE_TOKEN"))

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    top_p=0.9
)

def generate_answer(user_message):
    messages = [
        {"role": "user", "content": user_message},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )

    out_content = outputs[0]["generated_text"][-1]['content']
    try:
        response = json.loads(out_content)
    except json.JSONDecodeError:
        # Remove code block markdown (generated response is not json_obj but text)
        try:
            response = out_content.strip("```").strip("json\n")
            response = json.loads(response)
        except:
            response = None
    except:
        response = None

    if response is None:
        print(f"Failed to parse {out_content}")

    return response


def try_generate_answer(user_message):
    max_repeats = 10
    for i in range(max_repeats):
        answer = generate_answer(user_message)
        if answer is not None:
            return answer
        print(f"retrying {i}/{max_repeats}")


def create_message(query, passage):
    return f""""
    You will be presented with a query and passage. Your task is 
    to extract passage spans which are relevant to the query.
    Span must be comprehensive: removing the span from the text should make it irrelevant 
    to the query.
    Span must be plausible: human reading only this span should be convinced that text is relevant.   
    
    You may select multiple spans if needed, but ensure that the selected sections do not overlap. 
    Try not to select entire sentences, but only fine-grained spans.
    Do not correct or modify the text!
    Include all grammatical and syntactic errors from the original text, 
    do not remove senseless spaces or punctuation.
    
    Return only json_object with key 'spans' and list of selected spans 
    (text, start, end) as value. 
    \n
    Query: {query}
    Passage: {passage}
    """


# Load the dataset
dataset = load_dataset("bclavie/msmarco-2m-triplets")

# Display a sample from the dataset
out_dataset = []
nr_rows = len(dataset['train'])
nr_rows = 100_000
skipped = 0
for i in tqdm(range(nr_rows)):
    data = dataset['train'][i]
    query = data["query"]
    passage = data["positive"]

    ans = try_generate_answer(create_message(query, passage))
    if ans is None:
        skipped += 1
        continue

    out = {
        'query': data['query'],
        'positive': data['positive'],
        'negative': data['negative'],
        'selected_spans': ans["spans"]
    }
    out_dataset.append(out)

    with open("out.json", "w") as write_file:
        json.dump(out_dataset, write_file, ensure_ascii=False, indent=4)

print(f"problem with parsing at {skipped}/{nr_rows}")
# 0. prepare data
# try to optimize message (comprehensiveness, plausibility, maybe just one span?)
# download json data from .cache folder
# split it to two halves, make each gpu process the corresponding half
# stop using transformers pipeline
# process output: tokenize texts and prepare 0/1 label vectors

# 1. get scores from colbert
# - compare for plausibility

# 2. distill scores to colbert
# optimize BCE for labels
# KD ????? for normal scores
# z_t - z_{t-1} loss for continuity
# |z| for sparsity
