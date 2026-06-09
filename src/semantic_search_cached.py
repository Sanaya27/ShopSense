import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

print("Loading products...")

products = pd.read_csv(
    "data/expanded_products.csv"
)

print("Loading saved embeddings...")

product_embeddings = joblib.load(
    "models/product_embeddings.pkl"
)

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

query = input("\nSearch for a product: ")

query_embedding = model.encode(
    query,
    convert_to_tensor=True
)

similarities = util.cos_sim(
    query_embedding,
    product_embeddings
)[0]

products["score"] = similarities.cpu().numpy()

results = products.sort_values(
    by="score",
    ascending=False
)

print("\nTop Results:\n")

for _, row in results.head(5).iterrows():
    print(
        f"{row['title']} | Score: {row['score']:.3f}"
    )