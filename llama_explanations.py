import json
from os import write

import transformers
import torch
from datasets import load_dataset
from huggingface_hub import login
import os

from tqdm import tqdm

login(token=os.getenv("HUGGINGFACE_TOKEN"))

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
    You will be presented with a query and passage, please 
    extract passage spans which are most relevant to the query.
    
    Span must be comprehensive: removing the span from the text should make it irrelevant 
    to the query.
    Span must be plausible: human reading only this span should be convinced that text is relevant.   
    Let's read both passage and query, and then carefully consider
    relevance of each passage part to the query. 
    
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


if __name__ == "__main__":

    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        top_p=0.9
    )


    # Load the dataset
    dataset_file = "data/29_random_samples.jsonl"
    write_path = f"data/29_random_samples_{model_id.split('/')[1]}.jsonl"

    # Display a sample from the dataset
    with open(dataset_file, 'r') as f:
        data = [json.loads(line) for line in f]

    skipped = 0
    out_dataset = []
    for d in data:
        query = d["query"]
        passage = d["positive"]

        ans = try_generate_answer(create_message(query, passage))
        if ans is None:
            skipped += 1
            continue

        out = {
            'query': d['query'],
            'positive': d['positive'],
            'negative': d['negative'],
            'selected_spans': ans["spans"]
        }
        out_dataset.append(out)

        with open(write_path, "w") as write_file:
            for out_d in out_dataset:
                write_file.write(json.dumps(out_d, ensure_ascii=False) + "\n")

    print(f"problem with parsing at {skipped}/{len(data)}")


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
