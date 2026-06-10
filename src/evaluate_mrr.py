import pandas as pd
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

queries = (
    examples[["query", "product_id"]]
    .drop_duplicates()
    .sample(100, random_state=42)
)

reciprocal_ranks = []

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

    rank = None

    for i, product_id in enumerate(
        retrieved_products,
        start=1
    ):
        if product_id == true_product:
            rank = i
            break

    if rank:
        reciprocal_ranks.append(
            1 / rank
        )
    else:
        reciprocal_ranks.append(
            0
        )

mrr = sum(reciprocal_ranks) / len(
    reciprocal_ranks
)

print(
    f"\nMRR@10: {mrr:.4f}"
)