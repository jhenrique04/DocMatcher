import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class Vectorizer:
    def __init__(self, model_path):
        self.model = KeyedVectors.load_word2vec_format(model_path, unicode_errors='ignore')

    def calculate_document_vector(self, processed_text):
        vectors = [self.model[word] for word in processed_text if word in self.model.key_to_index]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.model.vector_size)

    def calculate_similarity(self, doc_vector, all_vectors):
        similarities = cosine_similarity([doc_vector], all_vectors)
        return similarities[0]