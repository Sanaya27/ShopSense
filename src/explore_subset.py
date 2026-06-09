import pandas as pd

products = pd.read_csv(
    "data/shopsense_products.csv"
)

examples = pd.read_csv(
    "data/shopsense_examples.csv"
)

print("\nProducts:")
print(products.shape)

print("\nExamples:")
print(examples.shape)

print("\nUnique Queries:")
print(examples["query"].nunique())

print("\nSample Queries:")
print(
    examples["query"]
    .drop_duplicates()
    .sample(10)
    .tolist()
)