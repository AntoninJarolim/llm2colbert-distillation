import argparse
import json
import os
from collections import defaultdict

import jsonlines
from boltons.iterutils import unique
from numpy.ma.core import max_filler
from tqdm import tqdm
from transformers import AutoTokenizer

from ftfy import fix_text
import text_utils
import random
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid")


def extract_ids_to_extract_relevancy_for(input_file="colbert_data/training/examples.json",
                                         out_selected_examples="colbert_data/training/examples_800k_unique_selected.jsonl",
                                         out_generation_pairs_file="data/input/64_way/examples_800k_unique.jsonl"):
    out_generate = {}
    out_generate_examples = {}
    ranks = defaultdict(int)
    nr_debug_prints = 1
    nr_have_relevant_not_in_batch = 0
    with jsonlines.open(input_file) as reader:
        for example_index, reranked_passages in tqdm(enumerate(reader),
                                                     desc="Processing lines",
                                                     unit="lines",
                                                     total=nr_reranked_examples):
            q_id = reranked_passages[0]

            # Extract ids and scores to separate lists from a structure
            batch_psgs, batch_scores = zip(*reranked_passages[1:])
            # Filter out psg_ids that are annotated relevant in ms marco
            batch_ms_marco_relevant = [psg_id for psg_id in batch_psgs if psg_id in qrels[q_id]]

            if len(batch_ms_marco_relevant) > 0:
                psg_type = 0  # Type 0: From ms marco
                psg_id = batch_ms_marco_relevant[0]
                ranks[batch_psgs.index(psg_id)] += 1
                # print(batch_docs.index(psg_id))
            else:
                psg_type = 1  # Type 1: Highest retrieved passage
                psg_id = reranked_passages[1][0]

            # Just extract passage text from ms-marco collection
            psg_text = collection[psg_id]

            if q_id not in out_generate or out_generate[q_id]['psg_type'] == 1:
                out_generate_examples[q_id] = reranked_passages
                out_generate[q_id] = (
                    {
                        "q_id": q_id,
                        "q_text": queries[q_id],
                        "psg_id": psg_id,
                        "psg_text": psg_text,
                        "psg_type": psg_type
                    }
                )

            # Check first one highest score, if first not annotated as relevant
            if psg_type == 1:
                if not all(batch_scores[0] >= score for score in batch_scores[1:]):
                    print(batch_scores[0], max(batch_scores[1:]))
                    exit(999)

            # stats/debugging
            if psg_type == 1 and len(qrels[q_id]) > 0:
                nr_have_relevant_not_in_batch += 1
                if nr_debug_prints > 0:
                    debug_print(batch_psgs, psg_id, q_id)
                    nr_debug_prints -= 1

    with jsonlines.open(out_generation_pairs_file, "w") as writer:
        writer.write_all(out_generate.values())

    with jsonlines.open(out_selected_examples, "w") as writer:
        writer.write_all(out_generate_examples.values())

    # Compute stats
    type_counter = defaultdict(int)
    for out in out_generate.values():
        type_counter[out['psg_type']] += 1

    for k, v in type_counter.items():
        type_str = "ms-marco" if k == 0 else "retrieved"
        print(f"Type {type_str} count: {v}")

    weighted_sum = 0
    for rank, nr_at_rank in ranks.items():
        rank = rank + 1
        weighted_sum += rank * nr_at_rank
        print(f"Rank {rank}: {nr_at_rank / sum(ranks.values()):.2f}")

    print(f"Average rank of annotated passage: {weighted_sum / sum(ranks.values())}")

    print(f"Nr of queries with annotated passage not in batch: {nr_have_relevant_not_in_batch}")


def debug_print(batch_psgs, psg_id, q_id):
    print(f"Passage with id {psg_id} has annotated relevant passage with ids {qrels[q_id]}")
    print(f"In batch, ids({len(batch_psgs)}) are: {batch_psgs}")
    print("\tQuery text:")
    print(queries[q_id])
    print("\tHighest retrieved passages (rank: text):")
    for i in range(10):
        print(f"{i}: {collection[batch_psgs[i]]}")
    print(f"\tAnnotated passage texts ({len(qrels[q_id])}):")
    for psg_id in qrels[q_id]:
        print(collection[psg_id])


# def unify_triplets_output(triplets_explained='data/triplets_explained.jsonl',
#                           generate_relevancy_ids="data/input/64_way/examples_800k_unique.jsonl",
#                           unified_out_file='data/extracted_relevancy_outs/triplets_explained_unified.jsonl',
#                           write_all=False):
def unify_triplets_output(triplets_explained='data/35_sample_dataset/35_random_samples_explained.jsonl',
                          generate_relevancy_ids="data/input/64_way/examples_800k_unique.jsonl",
                          unified_out_file='data/extracted_relevancy_outs/35_sample_dataset/35_explained_unified.jsonl',
                          write_all=True
                          ):
    # q_ids -> generate
    generate_for_id = {}
    with jsonlines.open(generate_relevancy_ids) as reader:
        for generate in reader:
            generate_for_id[generate['q_id']] = generate['psg_id']

    reverse_search_queries = {}
    for q_id, q in queries.items():
        reverse_search_queries[q] = q_id

    reverse_search_collection = {}
    for p_id, p in collection.items():
        reverse_search_collection[p] = p_id

    done_counter = 0

    out_data = []
    with jsonlines.open(triplets_explained) as reader:
        for triplet in tqdm(reader, desc="Processing lines", unit="lines", total=nr_in_one_experiment):
            q_id = reverse_search_queries[triplet['query']]
            p_id = reverse_search_collection[triplet['positive']]

            if write_all or generate_for_id[q_id] == p_id:
                done_counter += 1

                out_data.append(
                    {
                        "q_id": q_id,
                        "q_text": triplet['query'],
                        "psg_id": p_id,
                        "psg_text": triplet['positive'],
                        "psg_type": 0,
                        "selected_spans": triplet['selected_spans']
                    }
                )

    with jsonlines.open(unified_out_file, mode="w") as writer:
        writer.write_all(out_data)

    print(f"Done counter: {done_counter}")


def write_tsv_line(relevancy_out_f, q_id, psg_id, psg_type, span_list):
    # tsv out file format: `<q_id> <psg_id> <psg_type> <ERS1> <ERS2> ...`
    # ERS - extracted relevancy string
    out_list = [q_id, psg_id, psg_type, *span_list]
    relevancy_out_f.write(
        '\t'.join([str(obj) for obj in out_list]) + "\n"
    )


def validate_out_tsv(relevancy_out_path):
    collection = load_collection()

    with open(relevancy_out_path, mode='r') as relevancy_out_f:
        for line in relevancy_out_f:
            q_id, psg_id, psg_type, *span_list = line.strip().split('\t')
            q_id, psg_id, psg_type = int(q_id), int(psg_id), int(psg_type)

            assert psg_type >= 0, f"psg_type is negative: {psg_type}"
            if len(span_list) <= 0:
                raise AssertionError(f"No spans selected {q_id}")

            passage_text = collection[psg_id]
            text_utils.find_spans(passage_text, span_list)

    print("Validation of out tsv passed.")


def find_success(text, spans):
    # Ensure that the spans are not valid at the start
    try:
        text_utils.find_spans(text, spans)
        return True
    except AssertionError:
        return False


def two_spans_diff(span1, span2):
    return sum(c1 != c2 for c1, c2 in zip(span1, span2)) + abs(len(span1) - len(span2))


def heuristic_finding(text, span):
    """Tries to modify as little as possible chars in span_text so it is in text after modification"""
    span_start = text_utils.find_span_start_index(text.lower(), span.lower())
    if span_start == -1:
        # print(f"\t\t text: '{text}'")
        # print(f"\t\t span: '{span}'")
        # print()
        return span

    found_text_span = text[span_start:span_start + len(span)]
    diff_chars = two_spans_diff(found_text_span, span)
    if diff_chars > 3:
        print(f"\n\t text: '{text}'")
        print(f"\t\t modified({diff_chars}):")
        print(f"\t\t\t '{span}' to")
        print(f"\t\t\t '{found_text_span}' ")

    if diff_chars <= 10:
        return found_text_span
    return span


def try_find_spans(text, spans):
    # Ensure that the spans are not valid at the start
    assert not find_success(text, spans)

    # Try to find the spans with modified text
    modified_spans = []
    for span in spans:
        if not find_success(text, [span]):
            # Modify span if not found directly
            span['text'] = heuristic_finding(text, span['text'])
        modified_spans.append(span)

    # Return modified spans in case of success
    if find_success(text, modified_spans):
        return modified_spans
    return None


def create_out_tsv_human(
        # for_dataset='ms-marco-human-explained'
        for_dataset='accuracy_dataset'
):
    if for_dataset == 'ms-marco-human-explained':
        generation_out_dir = 'data/extracted_relevancy_outs/ms-marco-human-explained'
        relevancy_out_path = 'data/extracted_relevancy_ms-marco-human-explained.tsv'
    elif for_dataset == 'accuracy_dataset':
        generation_out_dir = 'data/extracted_relevancy_outs/accuracy_dataset'
        relevancy_out_path = 'data/extracted_relevancy_accuracy_dataset.tsv'
    else:
        raise ValueError("Unknown dataset")

    merge_datafile_name = 'out_all_explained.jsonl'
    shared_data_counter = 0
    collection_queries = {}
    collection_passages = {}
    tsv_out = defaultdict(list)

    for generation_out_file in os.listdir(generation_out_dir):
        if not generation_out_file.endswith(".jsonl"):
            print(f"Skipping {generation_out_file}")
            continue

        if generation_out_file == merge_datafile_name:
            continue

        out_file_path = os.path.join(generation_out_dir, generation_out_file)
        # Read the generated relevancy data in one file
        with jsonlines.open(out_file_path) as out_reader:
            for out_generated in out_reader:
                generated_key = out_generated['q_id'], out_generated['psg_id']

                collection_queries[out_generated['q_id']] = out_generated['q_id']  # out_generated['query']
                collection_passages[out_generated['psg_id']] = out_generated['psg_id']  # out_generated['passage']

                if generated_key in tsv_out:
                    shared_data_counter += 1

                for span in out_generated['selected_spans']:
                    if for_dataset == 'accuracy_dataset':
                        span = span['text']
                    tsv_out[generated_key].append(span)

                if len(out_generated['selected_spans']) == 0:
                    tsv_out[generated_key].append(out_generated['psg_text'][:50])

    with open(relevancy_out_path, mode='w') as relevancy_out_f:
        for (q_id, psg_id), span_list in tsv_out.items():
            write_tsv_line(relevancy_out_f, q_id, psg_id, 0, span_list)

    # Merge jsonl data so it can be written to jsonl file
    merged_jsonl_data = []
    for (q_id, psg_id), span_list in tsv_out.items():
        out_line = {
            'q_id': q_id,
            'query': collection_queries[q_id],
            'psg_id': psg_id,
            'passage': collection_passages[psg_id],
            'selected_spans': span_list
        }
        merged_jsonl_data.append(out_line)

    relevancy_out_json = os.path.join(generation_out_dir, merge_datafile_name)
    with jsonlines.open(relevancy_out_json, mode='w') as merged_jsonl:
        merged_jsonl.write_all(merged_jsonl_data)

    print(f"Shared number of data: {shared_data_counter}")
    print(f"Number of data in tsv: {len(tsv_out)}")

    validate_out_tsv(relevancy_out_path)


# def create_out_tsv(generation_out_dir='data/extracted_relevancy_outs/800k_unique',
#                    generate_relevancy_ids="data/input/64_way/examples_800k_unique.jsonl",
#                    relevancy_out_path='data/extracted_relevancy.tsv'):
def create_out_tsv(generation_out_dir='data/extracted_relevancy_outs/qrels.dev.small',
                   generate_relevancy_ids="data/input/qrels.dev.small/qrels.dev.small.jsonl",
                   relevancy_out_path='data/extracted_relevancy_qrels.dev.small.tsv-DELETE_ME'):
    # def create_out_tsv(generation_out_dir='data/extracted_relevancy_outs/35_sample_dataset',
    #                    generate_relevancy_ids=None,
    #                    relevancy_out_path='data/extracted_relevancy_35_sample_dataset.tsv'):
    # indexed by (q_id, psg_id) tuple
    tsv_out = {}
    if generate_relevancy_ids is not None:
        with jsonlines.open(generate_relevancy_ids) as reader:
            for generate in reader:
                q_id = generate['q_id']
                p_id = generate['psg_id']
                tsv_out[(q_id, p_id)] = (-1, None, None)

    generated_twice_counter = 0
    # Read the generated relevancy data in files a folder
    for generation_out_file in os.listdir(generation_out_dir):
        if not generation_out_file.endswith(".jsonl"):
            print(f"Skipping {generation_out_file}")
            continue

        out_file_path = os.path.join(generation_out_dir, generation_out_file)
        # Read the generated relevancy data in one file
        with jsonlines.open(out_file_path) as out_reader:
            for out_generated in out_reader:
                generated_key = out_generated['q_id'], out_generated['psg_id']

                # LLM generated twice (why?)
                if generate_relevancy_ids is not None and tsv_out[generated_key][0] != -1:
                    generated_twice_counter += 1
                    continue

                if 'extraction_error' in out_generated and out_generated['extraction_error']:
                    # LLM unable to select something which is in text
                    modified_spans = try_find_spans(out_generated['psg_text'], out_generated['selected_spans'])
                    if modified_spans:
                        out_generated['selected_spans'] = modified_spans
                        # psg_type == 2: from MS marco, fixed by heuristic
                        # psg_type == 3: top-1 retrieved, fixed by heuristic
                        psg_type = out_generated['psg_type'] + 2
                    else:
                        psg_type = -3
                elif out_generated['selected_spans'] is None:
                    # LLM unable to select something which is in text
                    # selected spans are not available for this query
                    psg_type = -3
                    out_generated['selected_spans'] = []
                elif not out_generated['selected_spans']:
                    # LLM selected nothing
                    psg_type = -2
                else:
                    psg_type = out_generated['psg_type']

                try:
                    spans_text = [span['text'] for span in out_generated['selected_spans']]
                except TypeError:
                    spans_text = out_generated['selected_spans']
                if psg_type >= 0 and not any([any([c.isalnum() for c in span]) for span in spans_text]):
                    # print(f"failed to find something alnum in this: {spans_text} for query {queries[out_generated['q_id']]}")
                    psg_type = -4

                tsv_out[generated_key] = (psg_type, spans_text, out_generated['psg_type'])

    error_counter = {0: defaultdict(int), 1: defaultdict(int)}
    with open(relevancy_out_path, mode='w') as relevancy_out_f:
        for (q_id, psg_id), (out_psg_type, span_list, in_psg_type) in tsv_out.items():
            error_counter[in_psg_type][out_psg_type] += 1

            if out_psg_type >= 0:
                write_tsv_line(relevancy_out_f, q_id, psg_id, out_psg_type, span_list)

    total_to_generate = error_counter[0][-1] + error_counter[1][-1]
    total_generated = len(tsv_out) - total_to_generate
    total_failed = error_counter[0][-3] + error_counter[1][-3]
    total_nothing = error_counter[0][-2] + error_counter[1][-2]

    any_ms_marco = sum(error_counter[0].values())
    any_top_one = sum(error_counter[1].values())

    print(f"Stats for {relevancy_out_path}")
    print(
        f"\t Generated + not yet generated / total:\t {total_generated:.0f}+{total_to_generate}/{nr_in_one_experiment:.0f}")
    print(f"\t err - generated twice: {generated_twice_counter}")
    print("\t err - not alphanumeric in selected spans: ", error_counter[0][-4] + error_counter[1][-4])

    print(f"\t err - LLM failed to generate correct extraction span: {total_failed}")
    if any_ms_marco > 0:
        print(
            f"\t\t ms-marco annotated: {error_counter[0][-3]}/{any_ms_marco} ({error_counter[0][-3] / any_ms_marco:.4f})")
    if any_top_one > 0:
        print(f"\t\t Top-1 retrieved: {error_counter[1][-3]}/{any_top_one} ({error_counter[1][-3] / any_top_one:.4f})")

    print(f"\t err - LLM selected nothing: {total_nothing}")
    if any_ms_marco > 0:
        print(
            f"\t\t ms-marco annotated: {error_counter[0][-2]}/{any_ms_marco} ({error_counter[0][-2] / any_ms_marco:.4f})")
    if any_top_one > 0:
        print(f"\t\t Top-1 retrieved: {error_counter[1][-2]}/{any_top_one} ({error_counter[1][-2] / any_top_one:.4f})")
    print()

    print(f"\t modified MS marco - {error_counter[0][2]}")
    print(f"\t modified top-1 retrieved - {error_counter[1][3]}")

    print()
    print(f"\t MS-marco annotated: {error_counter[0][0]}")
    print(f"\t Top-1 retrieved: {error_counter[1][1]}")
    correctly_generated = error_counter[0][0] + error_counter[1][1] + error_counter[0][2] + error_counter[1][3]
    print(f"\t \t-> correctly generated: {correctly_generated} / {total_generated} "
          f"({correctly_generated / total_generated:.4f})")

    validate_out_tsv(relevancy_out_path)


def add_tsv_to_examples(relevancy_out_path='data/extracted_relevancy.tsv',
                        examples_path="colbert_data/training/examples_800k_unique_selected.jsonl",
                        out_examples_path="colbert_data/training/examples_with_relevancy.jsonl"):
    examples = []
    with jsonlines.open(examples_path) as reader:
        for example in tqdm(reader, desc="loading examples", unit="lines", total=nr_in_one_experiment):
            examples.append(example)

    generated_for_q_id = defaultdict(list)
    with open(relevancy_out_path, mode='r') as relevancy_out_f:
        for line in tqdm(relevancy_out_f, desc="loading tsv", unit="lines", total=nr_in_one_experiment):
            q_id, psg_id, _, *span_list = line.strip().split('\t')
            q_id, psg_id = int(q_id), int(psg_id)

            generated_for_q_id[q_id].append(psg_id)

    print(f"Unique q_ids: {len(generated_for_q_id)}")

    new_examples = []
    for example in tqdm(examples, desc="Inserting into examples", unit="lines", total=nr_in_one_experiment):
        q_id = example[0]
        generated_psg_ids = generated_for_q_id[q_id]

        # Skip if no generated psg_ids
        if not generated_psg_ids:
            continue

        # Validate output
        batch_psgs, _ = zip(*example[1:])
        assert all(psg_id in batch_psgs for psg_id in generated_psg_ids), \
            "Not all generated psg_ids are in the batch"

        assert len(generated_psg_ids) == 1 and generated_psg_ids[0] == example[1][0]

        example.insert(1, generated_psg_ids)
        new_examples.append(example)

    with jsonlines.open(out_examples_path, mode="w") as writer:
        writer.write_all(new_examples)


def find_all_psg_ids_examples(examples_json):
    all_psg_ids = set()
    with jsonlines.open(examples_json) as reader:
        for example in tqdm(reader, desc="Loading examples", unit="lines", total=nr_in_one_experiment):
            batch_psgs, _ = zip(*example[2:])
            all_psg_ids.update(batch_psgs)
    return all_psg_ids


def create_collection_from_randon_samples(
        # for_dataset='35_sample_dataset'
        # for_dataset='ms-marco-human-explained'
        for_dataset='accuracy_dataset'
):
    samples = f'data/extracted_relevancy_outs/{for_dataset}'
    out_queries = f'colbert_data/evaluation/queries.eval.{for_dataset}.tsv'
    out_collection = f'colbert_data/evaluation/collection.{for_dataset}.tsv'
    out_collection_td = f'colbert_data/evaluation/collection.{for_dataset}.translate_dict.json'

    psg_key = 'passage' if for_dataset == 'ms-marco-human-explained' else 'psg_text'
    query_key = 'query' if for_dataset == 'ms-marco-human-explained' else 'q_text'

    data = []
    # for file in files open and load jsonl to data
    for file in os.listdir(samples):
        with jsonlines.open(os.path.join(samples, file)) as reader:
            for line in reader:
                data.append(line)

    new_collection = {}
    new_queries = {}

    for d in data:
        new_collection[d['psg_id']] = d[psg_key]
        new_queries[d['q_id']] = d[query_key]

    with open(out_queries, "w") as file:
        for q_id, q in new_queries.items():
            file.write(f"{q_id}\t{q}\n")

    write_new_collection(new_collection, out_collection, out_collection_td)
    print('done')


def create_collection_from_qrels(
        qrels='colbert_data/evaluation/qrels.dev.small.tsv',
        examples_json='colbert_data/training/examples_with_relevancy.jsonl',
        out_collection="colbert_data/evaluation/collection.dev.small_50-25-25.tsv",
        out_collection_translate_dict="colbert_data/evaluation/collection.dev.small_50-25-25.translate_dict.json",
):
    """
    Takes all passages from provided qrels and adds
    two times more data by adding:
        25% unseen in examples.json and
        25% seen in examples.json
    :return:
    """

    all_psg_ids = set(collection.keys())
    print(f"Total passages in collection: {len(all_psg_ids)}")

    examples_seen_ids = find_all_psg_ids_examples(examples_json)
    print(f"Total passages in examples: {len(examples_seen_ids)}")

    qrels_psg_ids = set()
    with open(qrels, "r") as file:
        for line in file:
            query_id, _, doc_id, _ = line.strip().split("\t")
            qrels_psg_ids.add(int(doc_id))

    dev_len = len(qrels_psg_ids)
    print(f"Total passages in qrels: {dev_len}")

    # 25% unseen in examples.json
    unseen_psg_ids = all_psg_ids - examples_seen_ids - qrels_psg_ids
    seen_psg_ids = examples_seen_ids - qrels_psg_ids
    print(f"Sampling 25% from unseen in examples.json: {len(unseen_psg_ids)}")
    print(f"Sampling 25% from seen in examples.json: {len(seen_psg_ids)}")

    # Sample 25% from unseen in examples.json
    unseen_psg_ids = list(unseen_psg_ids)
    seen_psg_ids = list(seen_psg_ids)

    random.seed(42)
    random.shuffle(unseen_psg_ids)
    random.shuffle(seen_psg_ids)

    unseen_psg_ids = unseen_psg_ids[:int(dev_len * 0.5)]
    seen_psg_ids = seen_psg_ids[:int(dev_len * 0.5)]

    assert set(unseen_psg_ids) & set(seen_psg_ids) & qrels_psg_ids == set()

    extrated_ids = unseen_psg_ids + seen_psg_ids + list(qrels_psg_ids)
    assert unique(extrated_ids)
    print(f"Total passages in new collection: {len(extrated_ids)}")
    assert len(extrated_ids) in [dev_len * 2 - 1, dev_len * 2, dev_len * 2 + 1], \
        f"{len(extrated_ids)} != {dev_len * 2}"

    # Write the new collection
    new_collection = {psg_id: collection[psg_id] for psg_id in extrated_ids}
    write_new_collection(new_collection, out_collection, out_collection_translate_dict)


def write_new_collection(new_collection_items, out_collection_path, out_trans_dict_path):
    translate_dict = {}

    with open(out_collection_path, "w") as file:
        for line_idx, (psg_id, psg) in enumerate(new_collection_items.items()):
            file.write(f"{line_idx}\t{psg}\n")
            translate_dict[line_idx] = psg_id

    json.dump(translate_dict, open(out_trans_dict_path, "w"), indent=2)


def tsv_to_jsonl_extracted(queries,
                           qrels_file='colbert_data/evaluation/qrels.dev.small.tsv',
                           out_generation_pairs_file="data/input/qrels.dev.small/qrels.dev.small.jsonl"):
    out_generate = {}
    with open(qrels_file, "r") as file:
        for line in file:
            query_id, _, doc_id, _ = line.strip().split("\t")
            query_id, doc_id = int(query_id), int(doc_id)

            out_generate[query_id] = (
                {
                    "q_id": query_id,
                    "q_text": queries[query_id],
                    "psg_id": doc_id,
                    "psg_text": collection[doc_id],
                    "psg_type": 0
                }
            )

    with jsonlines.open(out_generation_pairs_file, "w") as writer:
        writer.write_all(out_generate.values())

    print(f"Written {len(out_generate.values())} to {out_generation_pairs_file}")


def load_queries(queries_path="colbert_data/training/queries.train.tsv"):
    queries = {}
    with open(queries_path, "r") as q_file:
        for line in tqdm(q_file, desc="Loading queries", unit="lines", total=808731):
            q_id, q = line.strip().split("\t")
            queries[int(q_id)] = q
    return queries


def load_collection(collection_path="colbert_data/training/collection.tsv"):
    collection = defaultdict(str)
    with open(collection_path, "r") as coll_file:
        for line in tqdm(coll_file, desc="Loading collection", unit="lines", total=8841823):
            p_id, d = line.strip().split("\t")
            collection[int(p_id)] = d
    return collection


def load_qrels(q_rels_path="colbert_data/evaluation/qrels.train.tsv"):
    # Load ms marco GT data
    qrels = defaultdict(list)
    with open(q_rels_path, "r") as file:
        for line in file:
            query_id, _, doc_id, relevance_score = line.strip().split("\t")
            query_id = int(query_id)  # Convert QueryID to an integer
            doc_id = int(doc_id)  # Convert DocumentID to an integer
            relevance_score = int(relevance_score)  # Convert RelevanceScore to an integer

            if relevance_score > 0:
                qrels[query_id].append(doc_id)
    return qrels


def remove_no_extractions_from_qrels_and_queries(
        extracted_relevancy_data="data/extracted_relevancy_qrels.dev.small.tsv",
        qrels_data='colbert_data/qrels.dev.small.tsv',
        qrels_data_out='colbert_data/qrels.dev.small_ex_only.tsv',
        queries_data='colbert_data/queries.dev.small.tsv',
        queries_data_out='colbert_data/queries.dev.small_ex_only.tsv'
):
    """
    loads passage ids  with generated relevancy from extracted_relevancy_data
    and creates new file from qrels_data by writing only
    lines with psg_id in the extracted_relevancy_data
    :return:
    """

    psg_ids = set()
    query_ids = set()
    with open(extracted_relevancy_data, ) as reader:
        for line in reader:
            q_id, psg_id, *_ = line.strip().split("\t")
            psg_ids.add(int(psg_id))
            query_ids.add(int(q_id))

    lines_out = []
    with open(qrels_data, "r") as reader:
        for line in reader:
            _, _, psg_id, _ = line.strip().split("\t")
            if int(psg_id) in psg_ids:
                lines_out.append(line)

    with open(qrels_data_out, "w") as writer:
        writer.writelines(lines_out)

    print(f"Written {len(lines_out)} lines to {qrels_data_out}")

    # Write queries with only relevant passages
    lines_out = []
    with open(queries_data, "r") as reader:
        for line in reader:
            q_id, q = line.strip().split("\t")
            if int(q_id) in query_ids:
                lines_out.append(line)

    with open(queries_data_out, "w") as writer:
        writer.writelines(lines_out)

    print(f"Written {len(lines_out)} lines to {queries_data_out}")


def tsv_stats():
    tsv_outs = [
        'data/extracted_relevancy_800k_unique.tsv',
        'data/extracted_relevancy_35_sample_dataset.tsv',
        'data/extracted_relevancy_qrels.dev.small.tsv',
    ]

    max_spans = 8
    for file in tsv_outs:
        max = 0
        with open(file, 'r') as f:
            tab_counts = defaultdict(int)
            for line in f:
                count = line.count('\t') - 2  # - 2 for q_id and psg_id

                if count > max:
                    max = count

                count = count if count < max_spans else max_spans
                tab_counts[count] += 1

        print(f"\n=== Stats {file.split('/')[1]} ===")
        print(f"Max spans: {max}")
        for count in sorted(tab_counts):
            print(f"{tab_counts[count]} samples with {'' if count < max_spans else '>'}{count} spans")


def create_accuracy_dataset(output_file='data/accuracy_dataset/accuracy_dataset.jsonl'):
    """
    Creates a JSON Lines file with 60 entries:
    - 30 positive query-passage pairs
    - 30 negative query-passage pairs
    Format for each line:
    {"q_id": ..., "q_text": ..., "psg_id": ..., "psg_text": ...}
    """
    random.seed(42)

    # Select 30 queries that have at least one relevant passage
    eligible_q_ids = [q_id for q_id in qrels if qrels[q_id]]
    sampled_q_ids = random.sample(eligible_q_ids, 30)

    all_psg_ids = set(collection.keys())

    entries = []

    # Add 30 positive pairs
    for q_id in sampled_q_ids:
        q_text = queries[q_id]
        pos_psg_id = random.choice(qrels[q_id])
        pos_psg_text = collection[pos_psg_id]

        entries.append({
            "q_id": q_id,
            "q_text": q_text,
            "psg_id": pos_psg_id,
            "psg_text": pos_psg_text,
            "positive": True
        })

    # Add 30 negative pairs
    for q_id in sampled_q_ids:
        q_text = queries[q_id]
        pos_ids = set(qrels[q_id])

        neg_candidates = list(all_psg_ids - pos_ids)
        neg_psg_id = random.choice(neg_candidates)
        neg_psg_text = collection[neg_psg_id]

        entries.append({
            "q_id": q_id,
            "q_text": q_text,
            "psg_id": neg_psg_id,
            "psg_text": neg_psg_text,
            "positive": False
        })

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(entries)

    llm_input_file = "data/input/accuracy_dataset/accuracy_dataset.json"
    with jsonlines.open(llm_input_file, mode='w') as writer:
        writer.write_all(entries)


def enrich_accuracy_dataset_full(
        file_path='data/accuracy_dataset/accuracy_dataset.jsonl',
        out_file_path='data/accuracy_dataset/accuracy_dataset_full.jsonl',
):
    updated_entries = []
    updated_entries_starts = []

    # Read existing entries
    with jsonlines.open(file_path, mode='r') as reader:
        for entry in reader:
            entry['extraction'] = entry['psg_text']
            entry['source'] = 'full'
            updated_entries.append(entry)

            entry = entry.copy()
            entry['extraction'] = entry['psg_text'][:75]
            entry['source'] = 'start'
            updated_entries_starts.append(entry)

    # Write updated entries back
    with jsonlines.open(out_file_path, mode='w') as writer:
        writer.write_all(updated_entries)

    # Write updated entries but starts this time
    out_starts = out_file_path.replace('.jsonl', '_start.jsonl')
    with jsonlines.open(out_starts, mode='w') as writer:
        writer.write_all(updated_entries_starts)


def translate_psg_ids(
        translate_dict_path: str = "colbert_data/evaluation/collection.accuracy_dataset.translate_dict.json",
        input_jsonl: str = "data/other/accuracy_dataset_extraction_scores.jsonl",
        output_jsonl: str = "data/other/accuracy_dataset_extraction_scores_translated.jsonl"
):
    # Load the translation dictionary (keys are strings)
    with open(translate_dict_path, "r") as f:
        translate_dict = json.load(f)

    # Convert keys to integers for direct matching
    translate_dict = {int(k): v for k, v in translate_dict.items()}

    # Process the JSONL file and write the translated version
    with open(input_jsonl, "r") as infile, open(output_jsonl, "w") as outfile:
        for line in infile:
            record = json.loads(line)
            original_psg_id = record.get("psg_id")
            if original_psg_id in translate_dict:
                record["psg_id"] = translate_dict[original_psg_id]
            else:
                print(f"Warning: psg_id {original_psg_id} not found in translate_dict.")
            outfile.write(json.dumps(record) + "\n")


def apply_thresholded_extraction(
        thresholding_type,
        threshold,
        extraction_scores_path='data/other/accuracy_dataset_extraction_scores_translated.jsonl',
        dataset_path='data/accuracy_dataset/accuracy_dataset.jsonl',
        tokenizer_name="bert-base-uncased"
):
    output_path = f'data/accuracy_dataset/accuracy_dataset_{thresholding_type}.jsonl'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load extraction scores (psg_id and q_id as key tuple)
    extraction_scores = {}
    with open(extraction_scores_path, "r") as f:
        for line in f:
            item = json.loads(line)
            key = (item["psg_id"], item["q_id"])
            extraction_scores[key] = item

    with open(dataset_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            example = json.loads(line)
            key = (example["psg_id"], example["q_id"])

            if key not in extraction_scores:
                continue

            score_item = extraction_scores[key]
            psg_text = example["psg_text"]
            tokens = tokenizer.tokenize(psg_text)

            max_scores = score_item["max_scores_full"][2:-1]
            if len(tokens) != len(max_scores):
                raise ValueError(f"Token length mismatch for {key}: {len(tokens)} tokens vs {len(max_scores)} scores.")

            selected_tokens = [tok for tok, score in zip(tokens, max_scores)
                               if score is not None and score > threshold]

            # Detokenize: merge wordpieces (e.g., "inter", "##rel", "##ate" → "interrelate")
            detok_words = []
            current_word = ""
            for token in selected_tokens:
                if token.startswith("##"):
                    current_word += token[2:]
                else:
                    if current_word:
                        detok_words.append(current_word)
                    current_word = token
            if current_word:
                detok_words.append(current_word)

            example["extraction"] = " ".join(detok_words)
            example["source"] = thresholding_type

            fout.write(json.dumps(example) + "\n")


def convert_llm_selected_spans_to_extractions(
        input_path='data/extracted_relevancy_outs/accuracy_dataset/google~gemma-2-27b-it_from0-to60.jsonl',
        output_path='data/accuracy_dataset/accuracy_dataset_llm.jsonl',
):
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            example = json.loads(line)

            if "selected_spans" in example and isinstance(example["selected_spans"], list):
                # Sort spans by 'start'
                sorted_spans = sorted(example["selected_spans"], key=lambda x: x["start"])
                # Concatenate the 'text' fields
                extraction = " ".join(span["text"] for span in sorted_spans)
            else:
                extraction = ""

            # Add the new fields
            example["source"] = "LLM"
            example["extraction"] = extraction

            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")


def save_positive_samples(
        dataset_dir="data/accuracy_dataset",
        output_filename="data/accuracy_dataset/accuracy_dataset_all_sources_positive.jsonl"
):
    all_positive_samples = []

    excluded = ["accuracy_dataset.jsonl", "accuracy_dataset_all_sources_positive.jsonl"]
    for filename in os.listdir(dataset_dir):
        if filename in excluded or not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(dataset_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                if example.get("positive"):
                    all_positive_samples.append(example)

    with open(output_filename, "w", encoding="utf-8") as out_f:
        for example in all_positive_samples:
            for key in ["q_text", "psg_text", "extraction"]:
                if key in example:
                    example[key] = fix_text(example[key])
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")


def plot_extraction_lengths(
        dataset_dir="data/accuracy_dataset",
        tokenizer_name="bert-base-uncased",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    mean_lengths = {}

    excluded = ["accuracy_dataset.jsonl", "accuracy_dataset_all_sources_positive.jsonl"]

    for filename in os.listdir(dataset_dir):
        if filename in excluded or not filename.endswith(".jsonl"):
            continue

        label = filename.replace("accuracy_dataset_", "").replace(".jsonl", "")
        token_counts = []

        with open(os.path.join(dataset_dir, filename), "r") as f:
            for line in f:
                example = json.loads(line)
                if "extraction" not in example:
                    raise ValueError(f"Missing 'extraction' in {filename} for q_id {example['q_id']}")

                tokens = tokenizer.tokenize(example["extraction"])
                if example["positive"]:
                    token_counts.append(len(tokens))

        if token_counts:
            mean_lengths[label] = sum(token_counts) / len(token_counts)

    # Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(
        y=list(mean_lengths.keys()),
        x=list(mean_lengths.values()),
        palette="pastel",
        legend=False,
        hue=list(mean_lengths.keys())
    )
    plt.ylabel("Mean Number of Tokens in Extraction")
    plt.title(f"Average Extraction Length by File")
    plt.tight_layout()
    plt.show()
    plt.savefig("data/accuracy_dataset/extraction_lengths.png")


def distribute_for_annotation(
        input_path="data/accuracy_dataset/accuracy_dataset_all_sources_positive.jsonl",
        output_dir="data/accuracy_dataset/to_annotate",
):
    # set seed for reproducibility
    random.seed(42)

    os.makedirs(output_dir, exist_ok=True)

    # Load all entries and group by 'source'
    source_to_examples = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            source = ex.get("source")
            source_to_examples[source].append(ex)

    # Check and sort sources consistently
    sorted_sources = sorted(source_to_examples.keys())
    assert len(sorted_sources) == 5, f"Expected 5 sources, got {sorted_sources}"

    # Each source should have 30 examples
    for source, examples in source_to_examples.items():
        assert len(examples) == 30, f"Expected 30 examples per source, got {len(examples)} for {source}"

    # Sort by q_id for consistency
    sorted_sources = {k: sorted(v, key=lambda x: x["q_id"]) for k, v in source_to_examples.items()}

    # Prepare annotator assignments
    annotator_data = {i: [] for i in range(1, 6)}  # annotators 1–5

    # Assign 6 from each source to each annotator
    for j, source in enumerate(sorted_sources):
        examples = source_to_examples[source]
        for i in range(5):  # annotators 1 to 5
            start = (j * 6 + i * 6) % 30
            annotator_data[i + 1].extend(examples[start:start + 6])

    # Sanity checks
    for annotator_id, samples in annotator_data.items():
        assert len(samples) == 30, f"Annotator {annotator_id} has {len(samples)} samples, expected 24"

    # Save to files
    for i in range(1, 6):
        out_path = os.path.join(output_dir, f"{i}_annotator.jsonl")
        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            for ex in annotator_data[i]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # perform check on the output files
    for file in os.listdir(output_dir):
        unique_query_ids = set()
        source_dict = defaultdict(int)
        with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                q_id = example.get("q_id")
                unique_query_ids.add(q_id)
                source = example.get("source")
                source_dict[source] += 1

        assert len(unique_query_ids) == 30, \
            f"Expected 30 unique query IDs, got {len(unique_query_ids)} in {file}"
        assert all(count == 6 for count in source_dict.values()), \
            f"Expected 6 samples from each source, got {source_dict} in {file}"
        print()


def print_positive_extractions_by_source(
        filename='data/accuracy_dataset/accuracy_dataset_all_sources_positive.jsonl',
):
    """
    Load all .jsonl files in the specified directory, group entries by (q_id, psg_id),
    filter only positive examples, and print the query and extractions grouped by source.
    """
    grouped = defaultdict(list)

    # Load and aggregate all data
    with open(os.path.join(filename), "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("positive", False):
                key = (entry["q_id"], entry["psg_id"], entry["q_text"])
                grouped[key].append((entry["source"], entry["extraction"]))

    # Print grouped outputs
    for (q_id, psg_id, query), extractions in grouped.items():
        print(f"query: {query}")
        for source, extraction in extractions:
            print(f"{source}: {extraction}")
        print()  # blank line between entries


def arg_parse():
    parser = argparse.ArgumentParser(description="Command-line tool to perform various data operations.")

    # Flags for each function
    parser.add_argument("--extract-ids", action="store_true",
                        help="Extract IDs to generate relevancy for.")
    parser.add_argument("--unify-triplets-output",
                        action="store_true",
                        help="Add triplets to explained triplets data.")
    parser.add_argument("--compute-out-tsv-stats",
                        action="store_true",
                        help="Add triplets to explained triplets data.")
    parser.add_argument("--create-out-tsv", action="store_true",
                        help="Create output TSV files from generated relevancy data.")
    parser.add_argument("--add-tsv-to-examples", action="store_true",
                        help="Stores information about generated extraction scores into examples.json.")
    parser.add_argument("--create-collection-from-qrels", action="store_true",
                        help="Creates smaller collection from the collection.tsv file and queries small.")
    parser.add_argument("--tsv-to-jsonl-extracted", action="store_true",
                        help="Converts tsv file to jsonl file compatible for rationale extraction.")
    parser.add_argument("--remove-no-extractions-from-qrels", action="store_true",
                        help="Removes passages with not generated extractions from qrels."
                             "Used to remove some passages, for which LLM failed to generate correct extraction.")
    parser.add_argument("--create-collection-from-randon-samples", action="store_true",
                        help="Create collection (and translate dict) and queries, both tsv files from "
                             "a data in standard generated format. Used to generate new collection for 35 samples"
                             "dataset. Because this collection does not have qrels and queries.tsv files")
    parser.add_argument("--print-positive-extractions-by-source", action="store_true",
                        help="Prints positive extractions grouped by source.")

    parser.add_argument("--create-accuracy-dataset", action="store_true", )

    return parser.parse_args()


if __name__ == "__main__":
    # Following constants are used in some following functions
    file_path = "colbert_data/examples.json"
    nr_reranked_examples = 19409544  # from wc --lines examples.json
    nr_experiments = 24  # by counting nr of q_id in examples.json
    nr_in_one_experiment = nr_reranked_examples / nr_experiments

    args = arg_parse()
    load_data = (
            args.extract_ids or
            args.create_collection_from_qrels or
            args.tsv_to_jsonl_extracted or
            args.unify_triplets_output
    )

    load_data_dev = (
            args.create_accuracy_dataset and False
    )

    if load_data:
        qrels = load_qrels()
        queries = load_queries()
        collection = load_collection()

    if load_data_dev:
        dev_qrels_path = "colbert_data/evaluation/qrels.dev.small.tsv"
        dev_queries_path = "colbert_data/evaluation/queries.dev.small.tsv"
        # dev_collection_path = "colbert_data/evaluation/collection.dev.small_50-25-25.tsv"
        # using real collection beaucase dev has different ids
        qrels = load_qrels(dev_qrels_path)
        queries = load_queries(dev_queries_path)
        collection = load_collection()  # dev_collection_path)

    if args.extract_ids:
        extract_ids_to_extract_relevancy_for()
    if args.unify_triplets_output:
        unify_triplets_output()
    if args.create_out_tsv:
        # create_out_tsv()
        create_out_tsv_human()  # if you want human evaluated dataset (without errors etc.)
        # don't forget to call create_collection_from_randon_samples
        # create_collection_from_randon_samples()
    if args.add_tsv_to_examples:
        add_tsv_to_examples()
    if args.create_collection_from_qrels:
        create_collection_from_qrels()
    if args.create_collection_from_randon_samples:
        create_collection_from_randon_samples()
    if args.tsv_to_jsonl_extracted:
        dev_queries = load_queries("colbert_data/evaluation/queries.dev.small.tsv")
        tsv_to_jsonl_extracted(dev_queries)
    if args.remove_no_extractions_from_qrels:
        remove_no_extractions_from_qrels_and_queries()
    if args.compute_out_tsv_stats:
        tsv_stats()

    if args.create_accuracy_dataset:
        # create_accuracy_dataset()
        enrich_accuracy_dataset_full()
        # translate_psg_ids()
        convert_llm_selected_spans_to_extractions()
        thresholds = [
            ('dev_thresholded', 0.31041958621743976),
            ('35_tokens', -1.8),
        ]
        for output_path, threshold in thresholds:
            apply_thresholded_extraction(output_path, threshold)

        plot_extraction_lengths()
        save_positive_samples()
        distribute_for_annotation()

    if args.print_positive_extractions_by_source:
        print_positive_extractions_by_source()
