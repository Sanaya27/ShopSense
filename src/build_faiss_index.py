import joblib
import faiss
import numpy as np

print("Loading embeddings...")

embeddings = joblib.load(
    "models/product_embeddings.pkl"
)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

dimension = embeddings.shape[1]

print(f"Embedding Dimension: {dimension}")

index = faiss.IndexFlatIP(
    dimension
)

faiss.normalize_L2(
    embeddings
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    "models/products.faiss"
)

print(
    f"FAISS Index Created with {index.ntotal} products!"
)