from sentence_transformers import SentenceTransformer
from sentence_transformers import util

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

word1 = "earphones"
word2 = "running shoes"

embedding1 = model.encode(word1, convert_to_tensor=True)
embedding2 = model.encode(word2, convert_to_tensor=True)

similarity = util.cos_sim(embedding1, embedding2)

print(f"\nSimilarity between '{word1}' and '{word2}':")
print(similarity.item())