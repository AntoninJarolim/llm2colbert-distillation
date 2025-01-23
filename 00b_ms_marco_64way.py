import json
from collections import defaultdict

import jsonlines
from tqdm import tqdm

file_path = "colbert_data/qrels.train.tsv"
qrels = defaultdict(list)
with open(file_path, "r") as file:
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


def count_query_observations():
    q_counts = defaultdict(int)
    with open(file_path, "r") as file:
        for line_id, line in tqdm(enumerate(file), desc="Processing lines", unit="lines", total=nr_reranked_examples):
            if line_id == nr_in_one_experiment:
                break
            q_counts[q_id] += 1

            if q_id not in queries:
                print(f"{q_id} not in train queries set")
    return q_counts



def extract_ids_to_extract_relevancy_for(input_file="colbert_data/examples.json",
                                         out_file="colbert_data/examples_800k_unique.jsonl"):
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
                psg_type = 0 # Type 0: From ms marco
                psg_id = batch_ms_marco_relevant[0]
                ranks[batch_psgs.index(psg_id)] += 1
                # print(batch_docs.index(psg_id))
            else:
                psg_type = 1 # Type 1: Highest retrieved passage
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


extract_ids_to_extract_relevancy_for()

