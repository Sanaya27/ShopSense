import pandas as pd
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

print("Loading products...")

products = pd.read_csv(
    "data/shopsense_products_search.csv"
)

print("Loading FAISS index...")

index = faiss.read_index(
    "models/shopsense.faiss"
)

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

query = input("\nSearch Query: ")

query_embedding = model.encode(
    [query]
).astype("float32")

faiss.normalize_L2(
    query_embedding
)

k = 10

scores, indices = index.search(
    query_embedding,
    k
)

print("\nTop Results:\n")

for rank, idx in enumerate(indices[0]):
    product = products.iloc[idx]

    print(
        f"{rank+1}. "
        f"{product['product_title']}"
    )
    print(
        f"Score: {scores[0][rank]:.3f}"
    )
    print("-" * 60)