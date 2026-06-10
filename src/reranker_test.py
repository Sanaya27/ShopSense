from sentence_transformers import CrossEncoder

print("Loading reranker...")

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded successfully!")