import torch


class Analyzer:
    def euclidean_distance(self, embedding1, embedding2):
        """
        Calculate Euclidean distance between two embeddings.
        """
        return torch.dist(embedding1, embedding2, p=2).item()

    def average_distance_to_train_docs(self, test_embedding, train_embeddings):
        distances = [self.euclidean_distance(
            test_embedding, train_embedding) for train_embedding in train_embeddings]
        return sum(distances) / len(distances)

    def cosine_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        """
        return torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

    def analyze(self, embedding_test, embeddings_train):
        """
        Analyze a test document by comparing it with training documents.
        """
        similarities = [self.cosine_similarity(
            embedding_test, embedding_train) for embedding_train in embeddings_train]

        analysis_result = {
            "max_similarity": max(similarities),
            "min_similarity": min(similarities),
            "average_similarity": sum(similarities) / len(similarities)
        }
        return analysis_result
