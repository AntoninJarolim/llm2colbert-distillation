import json

from datasets import load_dataset, Dataset
import torch
from jsonlines import jsonlines
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


def load_msmarco():
    dataset = load_dataset("bclavie/msmarco-2m-triplets")["train"]

    queries = {x['query'] for x in dataset}
    positives = {x['positive'] for x in dataset}
    negatives = {x['negative'] for x in dataset}
    docs = positives.union(negatives)

    print(f"Number of docs: {len(docs)}")
    print(f"Number of queries: {len(queries)}")
    return list(docs), list(queries)


def token_overlap_score(doc_term_idxs, scores, terms, debug=True):
    term_scores = {}
    for idx in doc_term_idxs:  # non_zero_indices[1] gives column indices
        term_scores[terms[idx]] = scores[idx]

        # Sort by insertion
    term_scores = {k: v for k, v in sorted(term_scores.items(), key=lambda item: -item[1])}

    if debug:
        for k, v in term_scores.items():
            print(f"{term_scores[k]}: {k}")

    return term_scores


def score_passage_overlap(
        tokenizer,
        fitted_docs, fitted_queries,
        doc, query,
        terms):
    doc_transformed = fitted_docs.transform([doc])[0]
    diag_transformed = fitted_queries.transform([query])[0]

    scores = diag_transformed.toarray()[0] * doc_transformed.toarray()[0]
    overlap_idxs = scores.nonzero()[0]
    scores_nonzero = scores[overlap_idxs]

    overlap_dict = token_overlap_score(overlap_idxs, scores, terms, debug=False)

    tokenized_doc = tokenizer.tokenize(doc)
    scores_doc = torch.rand(len(tokenized_doc)) * 1e-6
    for i, t in enumerate(tokenized_doc):
        scores_doc[i] = overlap_dict.get(t, 0)

    return scores_doc


def main():
    # get dataset
    all_dialogs, all_queries = load_msmarco()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    vocab = tokenizer.get_vocab()

    ## Fit tf-idf
    vectorizer_queries = TfidfVectorizer(tokenizer=tokenizer.tokenize, vocabulary=vocab)
    fitted_queries = vectorizer_queries.fit(all_dialogs)
    # terms_queries = vectorizer_queries.get_feature_names_out()

    vectorizer_docs = TfidfVectorizer(tokenizer=tokenizer.tokenize, vocabulary=vocab)
    fitted_docs = vectorizer_docs.fit(all_queries)
    terms_docs = vectorizer_docs.get_feature_names_out()


    doc = 'My insurance ended so what should i do'
    query = 'ended insurance'

    scores = score_passage_overlap(
        tokenizer,
        fitted_docs, fitted_queries,
        doc, query,
        terms_docs)

    file_path = 'data/35_random_samples.jsonl'
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    for d in data:
        doc = d['positive']
        query = d['query']
        scores = score_passage_overlap(
            tokenizer,
            fitted_docs, fitted_queries,
            doc, query,
            terms_docs)
        d['score_mask'] = (scores > 0).tolist() # Convert to binary non-zero mask
        d['select_all'] = torch.ones_like(scores).to(dtype=torch.bool).tolist() # Add select-all baseline


    with jsonlines.open('data/35_random_samples_tf_idf.jsonl', mode='w') as writer:
        writer.write_all(data)

    # enumerate 35 dataset
    # convert scores to non-zero mask


if __name__ == "__main__":
    main()
