import pandas as pd

print("Loading datasets...")

examples = pd.read_parquet(
    "data/shopping_queries_dataset_examples.parquet"
)

products = pd.read_parquet(
    "data/shopping_queries_dataset_products.parquet"
)

print("Filtering US data...")

examples = examples[
    examples["product_locale"] == "us"
]

products = products[
    products["product_locale"] == "us"
]

print("Keeping only Exact and Substitute labels...")

examples = examples[
    examples["esci_label"].isin(["E", "S"])
]

print("Selecting 20,000 products...")

product_ids = (
    examples["product_id"]
    .drop_duplicates()
    .head(20000)
)

products_subset = products[
    products["product_id"].isin(product_ids)
]

examples_subset = examples[
    examples["product_id"].isin(product_ids)
]

print("Saving files...")

products_subset.to_csv(
    "data/shopsense_products.csv",
    index=False
)

examples_subset.to_csv(
    "data/shopsense_examples.csv",
    index=False
)

print("\nDone!")

print(
    f"Products: {len(products_subset)}"
)

print(
    f"Examples: {len(examples_subset)}"
)