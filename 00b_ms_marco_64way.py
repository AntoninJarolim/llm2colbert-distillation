import argparse
import os
from collections import defaultdict

import jsonlines
from boltons.iterutils import unique
from tqdm import tqdm
import text_utils
import random


def extract_ids_to_extract_relevancy_for(input_file="colbert_data/examples.json",
                                         out_selected_examples="colbert_data/examples_800k_unique_selected.jsonl",
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


def unify_triplets_output(triplets_explained='data/triplets_explained.jsonl',
                          generate_relevancy_ids="data/input/64_way/examples_800k_unique.jsonl"):
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

            if generate_for_id[q_id] == p_id:
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

    unified_out_file = 'data/extracted_relevancy_outs/triplets_explained_unified.jsonl'
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
            assert len(span_list) > 0, f"No spans selected {q_id}"

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
        print(f"\t\t text: '{text}'")
        print(f"\t\t span: '{span}'")
        print()
        return span

    found_text_span = text[span_start:span_start + len(span)]
    diff_chars = two_spans_diff(found_text_span, span)
    # print(f"\n\t text: '{text}'")
    # print(f"\t\t modified({diff_chars}):")
    # print(f"\t\t\t '{span}' to")
    # print(f"\t\t\t '{found_text_span}' ")

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


# def create_out_tsv(generation_out_dir='data/extracted_relevancy_outs',
#                    generate_relevancy_ids="data/input/64_way/examples_800k_unique.jsonl",
#                    relevancy_out_path='data/extracted_relevancy.tsv'):
def create_out_tsv(generation_out_dir='data/extracted_relevancy_outs/qrels.dev.small',
                   generate_relevancy_ids="data/input/qrels.dev.small/qrels.dev.small.jsonl",
                   relevancy_out_path='data/extracted_relevancy_qrels.dev.small.tsv'):
    # indexed by (q_id, psg_id) tuple
    tsv_out = {}
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
                if tsv_out[generated_key][0] != -1:
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

                spans_text = [span['text'] for span in out_generated['selected_spans']]
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
                        examples_path="colbert_data/examples_800k_unique_selected.jsonl",
                        out_examples_path="colbert_data/examples_with_relevancy.jsonl"):
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


def create_collection(qrels='colbert_data/qrels.dev.small.tsv',
                      examples_json='colbert_data/examples_with_relevancy.jsonl',
                      out_collection="colbert_data/collection.dev.small_50-25-25.tsv"):
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

    unseen_psg_ids = unseen_psg_ids[:int(len(qrels_psg_ids) * 0.25)]
    seen_psg_ids = seen_psg_ids[:int(len(qrels_psg_ids) * 0.25)]

    extrated_ids = unseen_psg_ids + seen_psg_ids + list(qrels_psg_ids)
    assert unique(extrated_ids)
    print(f"Total passages in new collection: {len(extrated_ids)}")

    # Write the new collection
    new_collection = [(psg_id, collection[psg_id]) for psg_id in extrated_ids]
    with open(out_collection, "w") as file:
        for psg_id, psg in new_collection:
            file.write(f"{psg_id}\t{psg}\n")


def tsv_to_jsonl_extracted(queries,
                           qrels_file='colbert_data/qrels.dev.small.tsv',
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


def arg_parse():
    parser = argparse.ArgumentParser(description="Command-line tool to perform various data operations.")

    # Flags for each function
    parser.add_argument("--extract-ids", action="store_true",
                        help="Extract IDs to generate relevancy for.")
    parser.add_argument("--unify-triplets-output",
                        action="store_true",
                        help="Add triplets to explained triplets data.")
    parser.add_argument("--create-out-tsv", action="store_true",
                        help="Create output TSV files from generated relevancy data.")
    parser.add_argument("--add-tsv-to-examples", action="store_true",
                        help="Stores information about generated extraction scores into examples.json.")
    parser.add_argument("--create-collection", action="store_true",
                        help="Creates smaller collection from the collection.tsv file and queries small.")
    parser.add_argument("--tsv-to-jsonl-extracted", action="store_true",
                        help="Converts tsv file to jsonl file compatible for rationale extraction.")
    parser.add_argument("--remove-no-extractions-from-qrels", action="store_true",
                        help="Removes passages with not generated extractions from qrels.")

    return parser.parse_args()


def load_queries(queries_path="colbert_data/queries.train.tsv"):
    queries = {}
    with open(queries_path, "r") as q_file:
        for line in tqdm(q_file, desc="Loading queries", unit="lines", total=808731):
            q_id, q = line.strip().split("\t")
            queries[int(q_id)] = q
    return queries


def load_collection():
    queries_path = "colbert_data/collection.tsv"
    collection = defaultdict(str)
    with open(queries_path, "r") as coll_file:
        for line in tqdm(coll_file, desc="Loading collection", unit="lines", total=8841823):
            p_id, d = line.strip().split("\t")
            collection[int(p_id)] = d
    return collection


def load_qrels():
    # Load ms marco GT data
    q_rels_path = "colbert_data/qrels.train.tsv"
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


if __name__ == "__main__":
    # Following constants are used in some following functions
    file_path = "colbert_data/examples.json"
    nr_reranked_examples = 19409544  # from wc --lines examples.json
    nr_experiments = 24  # by counting nr of q_id in examples.json
    nr_in_one_experiment = nr_reranked_examples / nr_experiments

    args = arg_parse()
    load_data = args.extract_ids or args.create_collection or args.tsv_to_jsonl_extracted

    if load_data:
        qrels = load_qrels()
        queries = load_queries()
        collection = load_collection()

    if args.extract_ids:
        extract_ids_to_extract_relevancy_for()
    if args.unify_triplets_output:
        unify_triplets_output()
    if args.create_out_tsv:
        create_out_tsv()
    if args.add_tsv_to_examples:
        add_tsv_to_examples()
    if args.create_collection:
        create_collection()
    if args.tsv_to_jsonl_extracted:
        dev_queries = load_queries("colbert_data/queries.dev.small.tsv")
        tsv_to_jsonl_extracted(dev_queries)
    if args.remove_no_extractions_from_qrels:
        remove_no_extractions_from_qrels_and_queries()
