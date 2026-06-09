import pandas as pd

# Load products
products = pd.read_csv("data/products.csv")

# User query
query = input("Search for a product: ").lower()

# Find matching products
results = products[
    products["title"].str.lower().str.contains(query, na=False)
]

print("\nSearch Results:\n")

if len(results) == 0:
    print("No products found.")
else:
    print(results[["product_id", "title"]])