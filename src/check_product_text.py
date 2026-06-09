import pandas as pd

products = pd.read_csv(
    "data/shopsense_products_search.csv"
)

print(
    products["search_text"].iloc[0]
)