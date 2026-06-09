import pandas as pd

examples = pd.read_parquet(
    "data/shopping_queries_dataset_examples.parquet"
)

products = pd.read_parquet(
    "data/shopping_queries_dataset_products.parquet"
)

us_examples = examples[
    examples["product_locale"] == "us"
]

us_products = products[
    products["product_locale"] == "us"
]

print("\nUS Examples:")
print(len(us_examples))

print("\nUS Products:")
print(len(us_products))

print("\nUnique Queries:")
print(us_examples["query"].nunique())

print("\nLabel Distribution:")
print(us_examples["esci_label"].value_counts())