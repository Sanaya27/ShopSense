import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

print("Loading data...")

products = pd.read_csv(
    "data/shopsense_products_search.csv"
)

examples = pd.read_csv(
    "data/shopsense_examples.csv"
)

index = faiss.read_index(
    "models/shopsense.faiss"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Use only a small evaluation sample
queries = (
    examples[["query", "product_id"]]
    .drop_duplicates()
    .sample(100, random_state=42)
)

hits = 0

for _, row in queries.iterrows():

    query = row["query"]
    true_product = row["product_id"]

    query_embedding = model.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        10
    )

    retrieved_products = (
        products.iloc[
            indices[0]
        ]["product_id"]
        .tolist()
    )

    if true_product in retrieved_products:
        hits += 1

recall_at_10 = hits / len(queries)

print(
    f"\nRecall@10: {recall_at_10:.4f}"
)