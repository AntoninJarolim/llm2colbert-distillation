import json
from cProfile import label
from tokenize import tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def tokenize_list(tokens, tokenizer):
    token_list = []
    for t in tokens.tolist():
        token_list.append(tokenizer.decode(t))
    return token_list


def find_span_start_index(text, span: str):
    """
    Find span sub-sequence in text and return text-wise indexes
    """
    if not type(span) is str:
        raise AssertionError("Invalid generated data structure.")

    text_len = len(text)
    span_len = len(span)

    # Loop through possible start indices in `text`
    for i in range(text_len - span_len + 1):
        # Check if the sub-sequence from `text` matches `span`
        if text[i:i + span_len] == span:
            return i  # Return the start index if a match is found

    return -1  # Return -1 if the span is not found in text


def find_spans(text, selected_spans):
    rationales = []

    if len(selected_spans) == 0:
        return rationales

    if not isinstance(selected_spans, list):
        print(selected_spans)
        raise AssertionError("Selected spans must be list!")

    start_find = 0
    if isinstance(selected_spans[0], dict):
        selected_spans = [span['text'] for span in selected_spans]
    for span in selected_spans:
        find_in_text = text[start_find:]
        span_start = find_span_start_index(find_in_text, span)
        if span_start == -1:
            raise AssertionError(f"Selected span '{span}' was not found in the text: {find_in_text}")
        span_length = len(span)
        rationales.append(
            {
                'start': start_find + span_start,
                'length': span_length,
                'end': start_find + span_start + span_length,
            }
        )
        start_find += span_start + span_length
    return rationales


class ExplanationsDataset(Dataset):

    def __init__(self, file_path, tokenizer, decode_positive_as_list=False):
        self.decode_positive_as_list = decode_positive_as_list
        self.tokenizer = tokenizer
        self.file_path = file_path
        # Load the entire JSONL file into memory
        with open(file_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Each sample has:
        'positive' passage and 'selected_spans', which needs to be converted to binary vector
        of relevance indications
        :param idx:
        :return:
        """
        sample: dict = self.data[idx]
        tokenized_positive, binary_rationales = self.tokenize_with_spans(sample['positive'], sample['selected_spans'])

        return {
            'query': sample['query'],
            'positive': sample['positive'],
            'negative': sample['negative'],
            'tokenized_negative': self.encode_text(sample['negative']),
            'tokenized_positive': tokenized_positive,
            'tokenized_positive_decoded': self.decode_list(tokenized_positive),
            'rationales': binary_rationales,
            'selected_spans': sample['selected_spans']
        }

    def encode_text(self, text):
        outputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        outputs.update({
            "offset_mapping": outputs["offset_mapping"].squeeze(0),
            "input_ids": outputs["input_ids"].squeeze(0),
        })
        return outputs

    def tokenize_with_spans(self, positive_text, selected_spans):
        # Find spans and extract starts and ends
        found_spans = find_spans(positive_text, selected_spans)
        spand_starts = torch.tensor([s['start'] for s in found_spans])
        spand_ends = torch.tensor([s['end'] for s in found_spans])

        # Encode and split offsets to starts and ends
        encoded = self.encode_text(positive_text)
        encoded_text = encoded["input_ids"]
        offset_mapping_starts = encoded["offset_mapping"][:, 0] # 0th dim are starts
        offset_mapping_ends = encoded["offset_mapping"][:, 1] # 1st dim are ends

        # Reshape for broadcasting each-to-each
        start_overlap = offset_mapping_starts[:, None] < spand_ends
        end_overlap = offset_mapping_ends[:, None] > spand_starts
        label_tensors = torch.any(start_overlap & end_overlap, dim=1)

        # Binary labels must match encoded text length in number of tokens
        assert len(encoded_text) == len(label_tensors)
        return encoded_text, label_tensors

    def decode_list(self, tokens):
        return (
            tokenize_list(tokens, self.tokenizer)
            if self.decode_positive_as_list
            else None
        )


if __name__ == "__main__":
    # Example usage
    # file_path = 'data/29_random_samples_explained.jsonl'
    file_path = 'data/29_random_samples_Meta-Llama-3.1-8B-Instruct.jsonl'
    # file_path = 'data/29_random_samples_gpt-4o-mini-2024-07-18.jsonl'
    # file_path = 'data/29_random_samples_gpt-4o-2024-08-06.jsonl'

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    dataset = ExplanationsDataset(file_path, tokenizer, decode_positive_as_list=True)
    print(dataset[1])

    # Iterate through the data
    for i in range(len(dataset)):
        d = dataset[i]
        print(d)
