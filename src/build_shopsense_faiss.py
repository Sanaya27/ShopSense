import joblib
import faiss
import numpy as np

print("Loading embeddings...")

embeddings = joblib.load(
    "models/shopsense_embeddings.pkl"
)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

print(
    f"Embeddings Shape: {embeddings.shape}"
)

faiss.normalize_L2(
    embeddings
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    "models/shopsense.faiss"
)

print(
    f"FAISS Index Created with {index.ntotal} products!"
)