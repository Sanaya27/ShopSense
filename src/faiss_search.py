import pandas as pd
import faiss
import numpy as np
import time

from sentence_transformers import SentenceTransformer

print("Loading products...")

products = pd.read_csv(
    "data/expanded_products.csv"
)

print("Loading FAISS index...")

index = faiss.read_index(
    "models/products.faiss"
)

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
start = time.time()
query = input("\nSearch for a product: ")

query_embedding = model.encode(
    [query]
).astype("float32")

faiss.normalize_L2(
    query_embedding
)

k = 5

scores, indices = index.search(
    query_embedding,
    k
)

print("\nTop Results:\n")

for rank, idx in enumerate(indices[0]):
    product = products.iloc[idx]

    print(
        f"{rank+1}. {product['title']} "
        f"| Score: {scores[0][rank]:.3f}"
    )
    
end = time.time()
print(
    f"\nSearch completed in "
    f"{(end-start)*1000:.2f} ms"
)