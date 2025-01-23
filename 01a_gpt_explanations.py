import argparse
import json
import os
import time
from datetime import datetime

from jsonlines import jsonlines
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from explainable_dataset import ExplanationsDataset


class OpenAIGenerator:
    def __init__(self, model_name, use_ollama=False):
        if use_ollama:
            print("Using ollama instead of openAI api!")
            self.client = OpenAI(
                base_url='http://athena20.fit.vutbr.cz:11434/v1',
                api_key='ollama',
            )
        else:
            self.client = OpenAI()
        self.model = model_name  # self.model = "gpt-4o-2024-08-06" "gpt-4o-mini"

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
        return generation_result


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


def create_batch_name(id_from, id_to, generated_data_dir):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    return f"{generated_data_dir}/{time_str}_batch-{id_from}-{id_to}.jsonl"


def create_batch_fix_name(fix_id, generated_data_dir):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    return f"{generated_data_dir}/{time_str}_batch_{fix_id}.jsonl"


def create_batch_file(data_chunk, api, jsonl_filename):
    """
    Create batch file for a range of ids in a jsonl format, which is suitable for OpenAI API.
    https://platform.openai.com/docs/guides/batch/getting-started
    """
    with jsonlines.open(jsonl_filename, "w") as task_writer:
        for row_id, d in data_chunk.items():
            message = messages_for_passages(d["query"], d["positive"], api)

            # Create sub-batch
            task = task_from_prompt(f"row_{row_id}", message)
            task_writer.write(task)

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
        # print(batch)
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


def sleep_with_progress(seconds, description=None):
    if description is None:
        description = "Waiting"

    for _ in tqdm(range(seconds), desc=description, unit="s"):
        time.sleep(1)


def read_input_data(file_path):
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


def get_all_responses(generated_data_dir, return_list=False):
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
    responses_out = {k: json.loads(v) for k, v in responses.items()}
    return [responses_out[row_id] for row_id in responses_out.keys()] if return_list else responses_out


def get_args():
    argparse.ArgumentParser(description='Generate explanations for MSMARCO dataset')
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip_generation", action='store_true',
                        help="only processes each output file")

    parser.add_argument('--output_data_name', type=str, required=True,
                        help="Name of final output file.")
    parser.add_argument('--input_data_name', type=str, required=True,
                        help="Name of input data file.")

    parser.add_argument("--model_name",
                        type=str, default="gpt-4o-2024-08-06",
                        help="model to use for generation, also used to select folder to process")
    parser.add_argument("--from_sample", type=int, default=0, help="Start index of the batch")
    parser.add_argument("--to_sample", type=int, default=30, help="End index of the batch")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Batching steps: range(-from-sample, --to-sample, --batch-step)")

    parser.add_argument("--use_ollama", action="store_true",
                        help="Instead of OpenAI api, local ollama will be used.")
    parser.add_argument("--force_rewrite", action="store_true",
                        help="Disables the check for generating into existing directory.")

    return parser.parse_args()


def generate_one_batch(data_chunk, generation_api, jsonl_filename, use_ollama):
    if use_ollama:
        generate_one_batch_ollama(data_chunk, generation_api, jsonl_filename)
    else:
        generate_one_batch_openai(data_chunk, generation_api, jsonl_filename)


def generate_one_batch_openai(data_chunk, generation_api, jsonl_filename):
    create_batch_file(data_chunk,
                      generation_api,
                      jsonl_filename
                      )
    batch_id = create_batch_job(jsonl_filename)
    download_output_batch(batch_id)

    sleep_with_progress(60,
                        description="Waiting before sending new batch file.")


def generate_one_batch_ollama(data_chunk, generation_api, jsonl_filename):
    responses = []
    for row_id, d in tqdm(data_chunk.items(), desc="Generating batch"):
        response = generation_api(create_message(d["query"], d["positive"]))
        choice = dict(response.choices[0])
        choice['message'] = dict(choice['message'])
        responses.append(
            {
                'custom_id': f"row_{row_id}",
                'response': {
                    'body': {
                        'choices': [choice]
                    }
                }
            }
        )
    out_filename = jsonl_filename.replace('.jsonl', '_output.jsonl')
    with jsonlines.open(out_filename, mode='w') as writer:
        writer.write_all(responses)


def generate_all_batches(data_chunks, generation_api, generated_data_dir, use_ollama):
    for data_chunk in data_chunks:
        loop_from = list(data_chunk.keys())[0]
        loop_to = list(data_chunk.keys())[-1]
        print(f"Processing {loop_from} to {loop_to}")
        jsonl_filename = create_batch_name(loop_from, loop_from + len(data_chunk), generated_data_dir)

        generate_one_batch(data_chunk, generation_api, jsonl_filename, use_ollama)


def generate_all_batches_fix(data_chunks, generation_api, generated_data_dir, use_ollama):
    for fix_id, data_chunk in enumerate(data_chunks):
        print(f"Creating {fix_id} fix batch file.")
        jsonl_filename = create_batch_fix_name(fix_id, generated_data_dir)
        generate_one_batch(data_chunk, generation_api, jsonl_filename, use_ollama)


def write_output(responses_out, output_data_file, input_data):
    with jsonlines.open(output_data_file, mode='w') as writer:
        for input_data, out_selected in zip(input_data, responses_out):
            writer.write(
                {
                    **input_data,
                    'selected_spans': out_selected['spans']
                }
            )


def read_out_data(output_data_file) -> list[dict]:
    out_data = []
    with jsonlines.open(output_data_file, mode='r') as reader:
        for line in reader:
            out_data.append(line)
    return out_data


def write_out_data(output_data_file, out_data):
    with jsonlines.open(output_data_file, mode='w') as writer:
        writer.write_all(out_data)


def update_output(responses_with_keys, output_data_file):
    out_data = read_out_data(output_data_file)

    for k, v in responses_with_keys.items():
        out_data[k]['selected_spans'] = v['spans']

    write_out_data(output_data_file, out_data)


def remove_rows_output(output_data_file, indexes_to_remove):
    out_data = read_out_data(output_data_file)

    for remove_idx in indexes_to_remove:
        out_data[remove_idx]['selected_spans'] = None

    write_out_data(output_data_file, out_data)


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


def prepare_out_dir(generated_data_dir, force_rewrite):
    try:
        os.makedirs(generated_data_dir)
    except FileExistsError:
        if not force_rewrite:
            raise AssertionError(f"Refused to generate data into existing directory: '{generated_data_dir}'.")


def dataset_improved(invalid_samples_history, max_regenerate_count, invalid_len):
    return (len(invalid_samples_history) < max_regenerate_count
            or not sum(invalid_samples_history[-max_regenerate_count:]) == max_regenerate_count * invalid_len)


def main():
    args = get_args()

    generation_api = OpenAIGenerator(args.model_name, use_ollama=args.use_ollama)
    generated_data_dir = os.path.join("data/generated_batches", args.model_name)
    input_data = read_input_data(args.input_data_name)

    if not args.skip_generation:
        prepare_out_dir(generated_data_dir, args.force_rewrite)
        data_chunks = [
            {j: input_data[j]
             for j in range(i, min(i + args.batch_size, len(input_data)))}
            for i in range(args.from_sample, args.to_sample, args.batch_size)
        ]
        generate_all_batches(data_chunks,
                             generation_api,
                             generated_data_dir,
                             args.use_ollama
                             )

    responses_out = get_all_responses(generated_data_dir, return_list=True)

    # Remove file if exists
    output_data_file = f"data/{args.output_data_name}"
    print(f"Saving output data to {output_data_file}")
    silent_remove(output_data_file)
    write_output(responses_out, output_data_file, input_data)

    last_fix = 0
    # Find all fix directories in the target directory
    for item in os.listdir(generated_data_dir):
        existing_fix_path = os.path.join(generated_data_dir, item)
        if os.path.isdir(existing_fix_path) and item.startswith('fix_'):
            responses_out = get_all_responses(existing_fix_path)
            update_output(responses_out, output_data_file)
            last_fix += 1

    invalid_samples_history = []
    max_regenerate_count = 5
    while (invalid_len := len(invalid_samples := find_invalid_samples(output_data_file))) > 0:
        print(f"Version after {last_fix} fixes has {invalid_len} invalid samples. "
              f"Trying to fix following indexes.")
        print(invalid_samples)

        fix_data_dir = os.path.join(generated_data_dir, f"fix_{last_fix}")
        os.makedirs(fix_data_dir)
        # get chunks of data but only for invalid indexes
        invalid_data_chunks = [
            {invalid_id: input_data[invalid_id] for invalid_id in invalid_samples[i:i + args.batch_size]}
            for i in range(0, len(invalid_samples), args.batch_size)
        ]
        generate_all_batches_fix(invalid_data_chunks,
                                 generation_api,
                                 fix_data_dir,
                                 args.use_ollama
                                 )

        # Update final output file with generated fixes
        responses_out = get_all_responses(fix_data_dir)
        update_output(responses_out, output_data_file)
        last_fix += 1

        # Exit loop if N numbers of tries did not improve dataset
        invalid_samples_history.append(invalid_len)
        if not dataset_improved(invalid_samples_history, max_regenerate_count, invalid_len):
            print(f"Exiting because the count of invalid samples "
                  f"was not changed in last {max_regenerate_count} iterations.")
            remove_rows_output(output_data_file, invalid_samples)
            print(f"Indexes {invalid_samples} were removed from output dataset before exiting.")
            break

    print(f"{last_fix} fixes applied, 0 invalid samples ")


if __name__ == "__main__":
    main()
