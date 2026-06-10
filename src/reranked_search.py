import pandas as pd
import faiss
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

print("Loading products...")

products = pd.read_csv(
    "data/shopsense_products_search.csv"
)

print("Loading FAISS index...")

index = faiss.read_index(
    "models/shopsense.faiss"
)

print("Loading retriever...")

retriever = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

query = input("\nSearch Query: ")

query_embedding = retriever.encode(
    [query]
).astype("float32")

faiss.normalize_L2(
    query_embedding
)

# Retrieve top 20
scores, indices = index.search(
    query_embedding,
    20
)

candidates = products.iloc[
    indices[0]
].copy()

pairs = []

for _, row in candidates.iterrows():

    pairs.append(
        [
            query,
            row["search_text"]
        ]
    )

rerank_scores = reranker.predict(
    pairs
)

candidates["rerank_score"] = rerank_scores

candidates = candidates.sort_values(
    by="rerank_score",
    ascending=False
)

print("\nTop Reranked Results:\n")

for rank, (_, row) in enumerate(
    candidates.head(10).iterrows()
):

    print(
        f"{rank+1}. {row['product_title']}"
    )

    print(
        f"Rerank Score: "
        f"{row['rerank_score']:.3f}"
    )

    print("-" * 60)