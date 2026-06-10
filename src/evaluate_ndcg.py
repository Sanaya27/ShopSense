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

# Sample 100 queries
query_sample = (
    examples["query"]
    .drop_duplicates()
    .sample(100, random_state=42)
)

ndcg_scores = []

for query in query_sample:

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

    # Relevant products for this query
    relevant = examples[
        examples["query"] == query
    ][["product_id", "esci_label"]]

    relevance_dict = {}

    for _, row in relevant.iterrows():

        if row["esci_label"] == "E":
            relevance_dict[row["product_id"]] = 2

        elif row["esci_label"] == "S":
            relevance_dict[row["product_id"]] = 1

    dcg = 0

    for rank, product_id in enumerate(
        retrieved_products,
        start=1
    ):

        rel = relevance_dict.get(
            product_id,
            0
        )

        dcg += (
            (2**rel - 1)
            / np.log2(rank + 1)
        )

    ideal_rels = sorted(
        relevance_dict.values(),
        reverse=True
    )[:10]

    idcg = 0

    for rank, rel in enumerate(
        ideal_rels,
        start=1
    ):

        idcg += (
            (2**rel - 1)
            / np.log2(rank + 1)
        )

    if idcg > 0:
        ndcg_scores.append(
            dcg / idcg
        )

ndcg = np.mean(
    ndcg_scores
)

print(
    f"\nNDCG@10: {ndcg:.4f}"
)