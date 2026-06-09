import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

print("Loading products...")

products = pd.read_csv(
    "data/expanded_products.csv"
)

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Generating embeddings...")

embeddings = model.encode(
    products["title"].tolist()
)

joblib.dump(
    embeddings,
    "models/product_embeddings.pkl"
)

print(
    f"Saved {len(embeddings)} embeddings successfully!"
)