#!/bin/bash


# Push input data to karolina
rsync -rz data/input/64_way xjarol06@karolina:~/projects/llm2colbert/data/input --info=progress2 --info=name0

# Push input - already generated to karolina
rsync -rz data/extracted_relevancy.tsv xjarol06@karolina:~/projects/llm2colbert/data/ --info=progress2 --info=name0

# Get output data from karolina

base_source="xjarol06@karolina:~/projects/llm2colbert/data/"
base_dest="./data/"

# List of folders to sync
folders=("generated_batches" "extracted_relevancy_outs")

# Loop through the folders and sync
for folder in "${folders[@]}"; do
    rsync -rz "${base_source}${folder}/" "${base_dest}${folder}" --info=progress2 --info=name0
done

