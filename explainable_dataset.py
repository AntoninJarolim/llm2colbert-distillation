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


def split_by_spans(text, spans):
    spans = sorted(spans, key=lambda x: x['start'])

    result = []
    current_pos = 0

    for span in spans:
        start = span['start']
        end = span['end']

        # Add any text before the current span
        if current_pos < start:
            result.append({
                'span': text[current_pos:start],
                'label': 0
            })

        # Add the specified span
        result.append(
            {
                'span': text[start:end],
                'label': 1
            }
        )

        # Update current position
        current_pos = end

    # Add any remaining text after the last span
    if current_pos < len(text):
        result.append({
            'span': text[current_pos:],
            'label': 0
        })

    return result


def find_span_start_index(self, text, span: str):
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
        sample = self.data[idx]
        selected_spans = sample['selected_spans']
        tokenized_positive, rationales = self.tokenize_with_spans(sample['positive'], selected_spans)

        return {
            'query': sample['query'],
            'positive': sample['positive'],
            'negative': sample['negative'],
            'tokenized_negative': self.encode_text(sample['negative']),
            'tokenized_positive': tokenized_positive,
            'tokenized_positive_decoded': self.decode_list(tokenized_positive),
            'rationales': rationales,
            'selected_spans': sample['selected_spans']
        }

    def find_spans(self, text, selected_spans):
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
            span_start = find_span_start_index(self, text[start_find:], span)
            if span_start == -1:
                continue
            span_length = len(span)
            # explanation[start_find + span_start:span_length] = 1
            rationales.append(
                {
                    'start': start_find + span_start,
                    'length': span_length,
                    'end': start_find + span_start + span_length,
                }
            )
            start_find += span_start + span_length
        return rationales

    def encode_text(self, text):
        if text == " ":
            # tokenizer discards just space
            if self.tokenizer.decode(self.tokenizer.encode(" ", add_special_tokens=False,)) == "":
                return torch.tensor([6])

        return self.tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).flatten()

    def tokenize_with_spans(self, text, span_strings):
        found_spans = self.find_spans(text, span_strings)
        splits = split_by_spans(text, found_spans)

        # Todo: probably better approach is to tokenize list as batch, and use lengths to create label tensors
        label_tensors = []
        encoded_spans = []
        for split in splits:
            encoded_span = self.encode_text(split['span'])
            label_tensor = torch.full(encoded_span.size(), split['label'])

            encoded_spans.append(encoded_span)
            label_tensors.append(label_tensor)

        encoded_spans = torch.cat(encoded_spans)
        label_tensors = torch.cat(label_tensors)

        assert len(encoded_spans) == len(label_tensors)
        return encoded_spans, label_tensors

    def decode_list(self, tokens):
        return (
            tokenize_list(tokens, self.tokenizer)
            if self.decode_positive_as_list
            else None
        )


if __name__ == "__main__":
    # Example usage
    # file_path = 'data/29_random_samples_explained.jsonl'
    # file_path = 'data/29_random_samples_Meta-Llama-3.1-8B-Instruct.jsonl'
    file_path = 'data/29_random_samples_gpt-4o-mini-2024-07-18.jsonl'
    # file_path = 'data/29_random_samples_gpt-4o-2024-08-06.jsonl'

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    dataset = ExplanationsDataset(file_path, tokenizer, decode_positive_as_list=True)
    print(dataset[1])

    # Iterate through the data
    for i in range(len(dataset)):
        d = dataset[i]
        print(d)
