import pandas as pd
import faiss

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

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

retriever = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

queries = (
    examples[["query", "product_id"]]
    .drop_duplicates()
    .sample(100, random_state=42)
)

hits = 0

for _, row in queries.iterrows():

    query = row["query"]
    true_product = row["product_id"]

    query_embedding = retriever.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        20
    )

    candidates = products.iloc[
        indices[0]
    ].copy()

    pairs = []

    for _, product_row in candidates.iterrows():

        pairs.append(
            [
                query,
                product_row["search_text"]
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

    retrieved_products = (
        candidates.head(10)["product_id"]
        .tolist()
    )

    if true_product in retrieved_products:
        hits += 1

recall_at_10 = hits / len(queries)

print(
    f"\nReranked Recall@10: "
    f"{recall_at_10:.4f}"
)