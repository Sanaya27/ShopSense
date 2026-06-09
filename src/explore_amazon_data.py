import pandas as pd

df = pd.read_parquet(
    "data/shopping_queries_dataset_examples.parquet"
)

print("Unique Queries:")
print(df["query"].nunique())

print("\nUnique Products:")
print(df["product_id"].nunique())
print("\nSample Queries:")
print(df["query"].sample(10).tolist())