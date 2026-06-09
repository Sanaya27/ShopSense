import pandas as pd

products = pd.read_parquet(
    "data/shopping_queries_dataset_products.parquet"
)

print(
    products["product_locale"]
    .value_counts()
)