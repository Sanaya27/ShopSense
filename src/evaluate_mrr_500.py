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

eval_queries = pd.read_csv(
    "data/eval_queries.csv"
)

index = faiss.read_index(
    "models/shopsense.faiss"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

rr_scores = []

for query in eval_queries["query"]:

    relevant_products = set(
        examples[
            examples["query"] == query
        ]["product_id"]
    )

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

    reciprocal_rank = 0

    for rank, product_id in enumerate(
        retrieved_products,
        start=1
    ):

        if product_id in relevant_products:

            reciprocal_rank = 1 / rank
            break

    rr_scores.append(
        reciprocal_rank
    )

mrr = sum(rr_scores) / len(rr_scores)

print(
    f"\nMRR@10 (500 Queries): {mrr:.4f}"
)