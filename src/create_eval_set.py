import pandas as pd

examples = pd.read_csv(
    "data/shopsense_examples.csv"
)

eval_queries = (
    examples["query"]
    .drop_duplicates()
    .sample(500, random_state=42)
)

eval_queries.to_csv(
    "data/eval_queries.csv",
    index=False
)

print(
    f"Saved {len(eval_queries)} evaluation queries."
)