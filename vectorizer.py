from gensim.models import KeyedVectors
import numpy as np

class Vectorizer:
    def __init__(self, word_embedding_path):
        self.model = KeyedVectors.load_word2vec_format(word_embedding_path)
    
    def calculate_document_vector(self, text):
        vectors = [self.model[word] for word in text if word in self.model.key_to_index]
        if not vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

    def calculate_similarity(self, doc_vector, other_vector):
        cosine_similarity = np.dot(doc_vector, other_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(other_vector))
        return cosine_similarity