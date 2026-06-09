import pandas as pd

products = pd.read_csv(
    "data/shopsense_products.csv"
)

products["search_text"] = (
    products["product_title"].fillna("")
    + " "
    + products["product_brand"].fillna("")
    + " "
    + products["product_description"].fillna("")
    + " "
    + products["product_bullet_point"].fillna("")
)

products.to_csv(
    "data/shopsense_products_search.csv",
    index=False
)

print("Done!")
print(products.shape)