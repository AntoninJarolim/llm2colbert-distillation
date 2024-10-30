from datasets import load_dataset

# Load the dataset
dataset = load_dataset("bclavie/msmarco-2m-triplets")

# Display a sample from the dataset
for i in range(5):
    print(dataset['train'][i])
