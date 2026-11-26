# llm2colbert-distillation
**Archived (Nov 2026).** The extraction pipeline continues in
[FGR-generator](https://github.com/AntoninJarolim/FGR-generator) and the analysis
notebooks were moved to `FGR-generator/notebooks/llm2colbert/`. Data that used to be
in `data/` is now at:

- `FGR-generator/data/extracted_spans/msmarco-legacy-gemma2/`: extracted spans, TSVs, generation inputs and experiment outputs
- `FGR-generator/data/raw_responses/msmarco-legacy-gemma2/`: raw per-batch LLM responses (`generated_batches/`)
- `explaining-ms-marco/data/relevance/`: accuracy-dataset annotator assignments and judgements
- `ColBERT/data/{training,evaluation}/`: final collections, queries and training TSVs
