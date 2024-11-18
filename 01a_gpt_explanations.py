import argparse
import json
import os
import time
from datetime import datetime
from math import inf
from os import write

from PIL.PdfParser import encode_text
from jsonlines import jsonlines
from numpy.distutils.fcompiler import failed_fcompilers
from openai import OpenAI
from pydantic.v1.errors import IterableError
from sympy.physics.units import action
from tqdm import tqdm
from transformers import AutoTokenizer

from explainable_dataset import ExplanationsDataset
from llama_explanations import create_message


class OpenAIGenerator:
    def __init__(self, model_name, use_ollama=False):
        if use_ollama:
            print("Using ollama instead of openAI api")
            self.client = OpenAI(
                base_url='http://athena20.fit.vutbr.cz:11434/v1',
                api_key='ollama',
            )
            self.model = 'ollama'
        else:
            self.client = OpenAI()
            self.model = model_name # self.model = "gpt-4o-2024-08-06" "gpt-4o-mini"

        self.temperature = 0.2
        self.max_tokens = 1024
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.system_message = "You are helpful linguistic specialist eager to complete given task."

    def create_message(self, prompt):
        return [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    def create_api_call_dict(self, message, from_prompt=True):
        """
        Create API call for OpenAI API
        :param message: Prompt for the API call or message generated with create_message()
        :param from_prompt: Use to transform user prompt to conversation format
        :return:
        """
        if from_prompt:
            message = self.create_message(message)

        return dict(
            model=self.model,
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            response_format={
                "type": "json_object"
            }
        )

    def __call__(self, message):
        api_dict = self.create_api_call_dict(message)
        generation_result = self.client.chat.completions.create(**api_dict)
        generated_answer = generation_result.choices[0].message.content
        return generated_answer


def turns_for_gpt(turns):
    gpt_diag_turns = []
    # remove last turn as it is the one generated from the agent
    for turn in turns[:-1]:
        gpt_diag_turns.append({
            "role": turn["role"],
            "turn_id": turn["turn_id"],
            "utterance": turn["utterance"]
        })
    return gpt_diag_turns


def task_from_prompt(custom_id, prompt):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": prompt
    }


def messages_for_passages(query, passage, openai_api):
    prompt_str = create_message(query, passage)
    api_message = openai_api.create_api_call_dict(prompt_str)
    return api_message


def init_writer(id_from, id_to, generated_data_dir):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    jsonl_filename = f"{generated_data_dir}/{time_str}_batch-{id_from}-{id_to}.jsonl"
    return jsonl_filename, jsonlines.Writer(open(jsonl_filename, "w"))


def create_batch_file(starting_id, data_chunk, api, generated_data_dir):
    """
    Create batch file for a range of ids in a jsonl format, which is suitable for OpenAI API.
    https://platform.openai.com/docs/guides/batch/getting-started
    """

    jsonl_filename, task_writer = init_writer(starting_id, starting_id + len(data_chunk), generated_data_dir)

    for i, d in tqdm(enumerate(data_chunk), desc="Dialogs"):
        message = messages_for_passages(d["query"], d["positive"], api)

        # Create sub-batch
        tasks = []
        task = task_from_prompt(f"row_{starting_id + i}", message)
        tasks.append(task)

        # Append tasks to the file
        task_writer.write_all(tasks)

    print(f"Batch file saved to {jsonl_filename}")
    return jsonl_filename


def create_batch_job(batch_filename):
    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(batch_filename, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"job on filename {batch_filename}",
            "for_filename": batch_filename
        }
    )

    print(f"Created batch ({batch.id}):")
    print(batch)
    return batch.id


def download_output_batch(batch_id):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    while batch.status != "completed":
        print(batch)
        print(f"Batch is not completed yet - status is {batch.status}")
        if batch.status == "failed":
            raise Exception(f"Batch failed: {batch}")
        sleep_with_progress(15,
                            description="Waiting for validation")
        batch = client.batches.retrieve(batch_id)

    print("Batch is completed")
    print(batch)
    result_file_id = batch.output_file_id
    result = client.files.content(result_file_id).content
    batch_filename = batch.metadata["for_filename"].replace('.jsonl', '_output.jsonl')
    with open(batch_filename, 'wb') as file:
        file.write(result)
    print(f"Output file saved to {batch_filename}")

    return batch_filename


def message_from_output(res):
    message_json = res['response']['body']['choices'][0]['message']['content']
    try:
        message = json.loads(message_json)
    except json.JSONDecodeError:
        # Remove code block markdown (generated response is not json_obj but text)
        message = message_json.strip("```").strip("json\n")
        message = json.loads(message)
    return message


def messages_from_file(batch_filename):
    messages = []
    with jsonlines.open(batch_filename) as reader:
        for res in reader:
            message = message_from_output(res)
            messages.append({
                "spans": message["spans"],
                "id": res["custom_id"]
            })
    return messages


def sleep_with_progress(seconds, description=None):
    if description is None:
        description = "Waiting"

    for _ in tqdm(range(seconds), desc=description, unit="s"):
        time.sleep(1)


def read_input_data(file_path='data/29_random_samples.jsonl'):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def silent_remove(output_data_file):
    try:
        os.remove(output_data_file)
    except OSError:
        pass


def process_output(process_file):
    responses = {}
    with open(process_file) as in_file:
        for line in in_file:
            parsed_data = json.loads(line)
            row_index = int(parsed_data['custom_id'].strip("row_"))
            content = parsed_data['response']['body']['choices'][0]['message']['content']
            responses[row_index] = content

    return responses


def get_all_responses(generated_data_dir):
    """
    :return: Sorted reposes based on custom id
    """
    # row_id -> index, content -> value
    responses = {}
    for filename in os.listdir(generated_data_dir):
        if filename.endswith("_output.jsonl"):
            process_filename = f"{generated_data_dir}/{filename}"
            print(f"Processing {process_filename}")
            responses.update(process_output(process_filename))

    # Create new list by sorting with keys - rowid, needed to match input
    responses_out = [responses[row_id] for row_id in responses.keys()]
    responses_out = [json.loads(r) for r in responses_out]
    return responses_out


def get_args():
    argparse.ArgumentParser(description='Generate explanations for MD2D dataset')
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip_generation", action='store_true',
                        help="only processes each output file")

    parser.add_argument("--model_name",
                        type=str, default="gpt-4o-2024-08-06",
                        help="model to use for generation, also used to select folder to process")
    parser.add_argument("--from_sample", type=int, default=0, help="Start index of the batch")
    parser.add_argument("--to_sample", type=int, default=1, help="End index of the batch")
    parser.add_argument("--batch_step", type=int, default=5,
                        help="Batching steps: range(-from-sample, --to-sample, --batch-step)")

    parser.add_argument("--use_ollama", action="store_true",
                        help="Instead of OpenAI api, local ollama will be used.")

    return parser.parse_args()


def generate_all_batches(from_sample, to_sample, batch_step, generation_api, generated_data_dir):
    input_data = read_input_data()
    for loop_from in range(from_sample, to_sample, batch_step):
        loop_to = loop_from + batch_step
        print(f"Processing {loop_from} to {loop_to}")

        batch_filename = create_batch_file(loop_from,
                                           input_data[loop_from:loop_to],
                                           generation_api,
                                           generated_data_dir
                                           )
        batch_id = create_batch_job(batch_filename)
        download_output_batch(batch_id)

        sleep_with_progress(60,
                            description="Waiting before sending new batch file.")


def write_output(responses_out, output_data_file):
    input_data = read_input_data()
    with jsonlines.open(output_data_file, mode='w') as writer:
        for input_data, out_selected in zip(input_data, responses_out):
            writer.write(
                {
                    **input_data,
                    'selected_spans': out_selected['spans']
                }
            )


def find_invalid_samples(output_data_file):

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    dataset = ExplanationsDataset(output_data_file, tokenizer, decode_positive_as_list=True)
    index = 0
    failed_indexes = []
    for i in range(len(dataset)):
        try:
            dataset[i]
        except AssertionError:
            failed_indexes.append(index)
        index += 1
    return failed_indexes



def main():
    args = get_args()

    generation_api = OpenAIGenerator(args.model_name, use_ollama=args.use_ollama)
    generated_data_dir = os.path.join("openAI_batches", args.model_name)
    os.makedirs(generated_data_dir, exist_ok=True)

    if not args.skip_generation:
        generate_all_batches(args.from_sample,
                             args.to_sample,
                             args.batch_step,
                             generation_api,
                             generated_data_dir
                             )

    responses_out = get_all_responses(generated_data_dir)

    # Remove file if exists
    output_data_file = f"data/29_random_samples_{args.model_name}.jsonl"
    print(f"Saving output data to {output_data_file}")
    silent_remove(output_data_file)
    write_output(responses_out, output_data_file)

    while (invalid_len := len(invalid_samples := find_invalid_samples(output_data_file))) > 0:
        print(f"Current version has {invalid_len} samples. Trying to fix following indexes.")
        print(invalid_samples)
        break

if __name__ == "__main__":
    main()
