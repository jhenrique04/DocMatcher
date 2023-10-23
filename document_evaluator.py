import numpy as np

class DocumentEvaluator:
    def __init__(self, vectorizer, all_docs_vectors):
        self.vectorizer = vectorizer
        self.all_docs_vectors = all_docs_vectors

    def evaluate_document(self, doc_vector):
        similarities = [self.vectorizer.calculate_similarity(doc_vector, other_vec) for other_vec in self.all_docs_vectors]
        continuous_score = np.mean(similarities) * 100
        return continuous_score