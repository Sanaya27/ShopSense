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

hits = 0
total = 0

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

    retrieved_products = set(
        products.iloc[
            indices[0]
        ]["product_id"]
    )

    if len(
        relevant_products.intersection(
            retrieved_products
        )
    ) > 0:
        hits += 1

    total += 1

recall_at_10 = hits / total

print(
    f"\nRecall@10 (500 Queries): {recall_at_10:.4f}"
)