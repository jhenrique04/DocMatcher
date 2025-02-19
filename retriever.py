from sentence_transformers import SentenceTransformer, util
import os

# Load a pre-trained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_reference_docs():
    """Load and preprocess reference cybersecurity documents."""
    reference_texts = []
    for file in os.listdir("reference_docs/"):
        if file.endswith(".txt"):
            with open(os.path.join("reference_docs/", file), "r", encoding="utf-8") as f:
                reference_texts.append(f.read())
    return reference_texts

def retrieve_relevant_docs(query, top_k=3):
    """Retrieve the most relevant cybersecurity references using embeddings."""
    reference_texts = load_reference_docs()
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    doc_embeddings = embedding_model.encode(reference_texts, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    top_indices = similarities.argsort(descending=True).tolist()[0][:top_k]

    relevant_docs = "\n\n".join([reference_texts[i] for i in top_indices])
    return relevant_docs
