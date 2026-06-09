import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

print("Loading products...")

products = pd.read_csv(
    "data/shopsense_products_search.csv"
)

print(f"Products: {len(products)}")

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Generating embeddings...")

embeddings = model.encode(
    products["search_text"].tolist(),
    show_progress_bar=True,
    batch_size=64
)

joblib.dump(
    embeddings,
    "models/shopsense_embeddings.pkl"
)

print(
    f"\nSaved {len(embeddings)} embeddings!"
)