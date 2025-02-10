import argparse
import os
from collections import defaultdict

import jsonlines
from tqdm import tqdm


def extract_ids_to_extract_relevancy_for(input_file="colbert_data/examples.json",
                                         out_file="data/input/64_way/examples_800k_unique.jsonl"):
    out_generate = {}
    ranks = defaultdict(int)
    nr_debug_prints = 1
    nr_have_relevant_not_in_batch = 0
    with jsonlines.open(input_file) as reader:
        for reranked_passages in tqdm(reader, desc="Processing lines", unit="lines", total=nr_reranked_examples):
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

    with jsonlines.open(out_file, "w") as writer:
        writer.write_all(out_generate.values())

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


def create_out_tsv(generation_out_dir='data/extracted_relevancy_outs',
                   generate_relevancy_ids="data/input/64_way/examples_800k_unique.jsonl",
                   relevancy_out_path='data/extracted_relevancy.tsv'):
    # indexed by (q_id, psg_id) tuple
    tsv_out = {}
    with jsonlines.open(generate_relevancy_ids) as reader:
        for generate in reader:
            q_id = generate['q_id']
            p_id = generate['psg_id']
            tsv_out[(q_id, p_id)] = (-1, None)

    no_relevance_generated = 0
    generated_twice = 0
    for generation_out_file in os.listdir(generation_out_dir):
        if not generation_out_file.endswith(".jsonl"):
            print(f"Skipping {generation_out_file}")

        out_file_path = os.path.join(generation_out_dir, generation_out_file)
        with jsonlines.open(out_file_path) as out_reader:
            for out_generated in out_reader:
                generated_key = out_generated['q_id'], out_generated['psg_id']
                if tsv_out[generated_key][0] != -1:
                    generated_twice += 1
                    # print(
                    #     f"Warning: generated twice: \n\tFirst:\n{out_generated}\n\n\tSecond:\n{tsv_out[generated_key]}"
                    # )

                # LLM unable to select something which is in text
                if out_generated['selected_spans'] is None:
                    no_relevance_generated += 1
                    continue

                # LLM selected nothing
                if not out_generated['selected_spans']:
                    no_relevance_generated += 1
                    continue

                spans_text = [span['text'] for span in out_generated['selected_spans']]
                tsv_out[generated_key] = (out_generated['psg_type'], spans_text)

    not_yet_generated = 0
    with open(relevancy_out_path, mode='w') as relevancy_out_f:
        for (q_id, psg_id), (psg_type, span_list) in tsv_out.items():

            # Already done
            if psg_type == -1:
                not_yet_generated += 1
                continue

            # tsv out file format: `<q_id> <psg_id> <psg_type> <ERS1> <ERS2> ...`
            # ERS - extracted relevancy string
            out_list = [q_id, psg_id, psg_type, *span_list]
            relevancy_out_f.write(
                '\t'.join([str(obj) for obj in out_list]) + "\n"
            )

    print(f"Generated {nr_in_one_experiment - not_yet_generated} / {nr_in_one_experiment}")
    print(f"To generate {not_yet_generated} / {nr_in_one_experiment}")
    print(f"Nr of no relevancy generated: {no_relevance_generated}")
    print(f"Generated twice: {generated_twice}")


def main():
    parser = argparse.ArgumentParser(description="Command-line tool to perform various data operations.")

    # Flags for each function
    parser.add_argument("--extract-ids", action="store_true",
                        help="Extract IDs to generate relevancy for.")
    parser.add_argument("--unify-triplets-output",
                        action="store_true",
                        help="Add triplets to explained triplets data.")
    parser.add_argument("--create-out-tsv", action="store_true",
                        help="Create output TSV files from generated relevancy data.")

    args = parser.parse_args()

    # Call functions based on the flags
    if args.extract_ids:
        extract_ids_to_extract_relevancy_for()
    if args.unify_triplets_output:
        unify_triplets_output()
    if args.create_out_tsv:
        create_out_tsv()


if __name__ == "__main__":
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

    queries_path = "colbert_data/queries.train.tsv"
    queries = {}
    with open(queries_path, "r") as q_file:
        for line in tqdm(q_file, desc="Loading queries", unit="lines", total=808731):
            q_id, q = line.strip().split("\t")
            queries[int(q_id)] = q

    queries_path = "colbert_data/collection.tsv"
    collection = defaultdict(str)
    with open(queries_path, "r") as coll_file:
        for line in tqdm(coll_file, desc="Loading collection", unit="lines", total=8841823):
            p_id, d = line.strip().split("\t")
            collection[int(p_id)] = d

    # Following constants are used in some following functions
    file_path = "colbert_data/examples.json"
    nr_reranked_examples = 19409544  # from wc --lines examples.json
    nr_experiments = 24  # by counting nr of q_id in examples.json
    nr_in_one_experiment = nr_reranked_examples / nr_experiments

    main()