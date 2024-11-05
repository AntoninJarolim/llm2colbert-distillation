import json
from cProfile import label
from tokenize import tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


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


def find_span_start_index(self, text, span):
    """
    Find span sub-sequence in text and return text-wise indexes
    """

    text_len = len(text)
    span_len = len(span)

    # Loop through possible start indices in `text`
    for i in range(text_len - span_len + 1):
        # Check if the sub-sequence from `text` matches `span`
        if text[i:i + span_len] == span:
            return i  # Return the start index if a match is found

    return -1  # Return -1 if the span is not found in text


class ExplanationsDataset(Dataset):

    def __init__(self, file_path, tokenizer):
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
            'rationales': rationales,
            'selected_spans': sample['selected_spans']
        }

    def find_spans(self, text, selected_spans):
        explanation = []

        start_find = 0
        for span in selected_spans:
            span_start = find_span_start_index(self, text[start_find:], span)
            span_length = len(span)
            # explanation[start_find + span_start:span_length] = 1
            explanation.append(
                {
                    'start': start_find + span_start,
                    'length': span_length,
                    'end': start_find + span_start + span_length,
                }
            )
            start_find += span_start + span_length
        return explanation

    def encode_text(self, text):
        return self.tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )[0]

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

# Example usage
file_path = 'data/30_random_samples_explained.jsonl'
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
dataset = ExplanationsDataset(file_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # shuffle=False to disable randomization

# Iterate through the data
for batch in dataloader:
    print(batch)
