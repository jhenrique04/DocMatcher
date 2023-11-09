from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

class DocumentVisualizer:
    def __init__(self):
        pass

    def calculate_similarity(self, new_vectors, vectors, labels, predicted_labels):
        highest_scores = []
        
        for i, new_vector in enumerate(new_vectors):
            predicted_label = predicted_labels[i]
            same_category_indices = np.where(labels == predicted_label)[0]
            same_category_vectors = vectors[same_category_indices]
            
            similarity_scores = cosine_similarity(new_vector.reshape(1, -1), same_category_vectors)
            highest_scores.append(np.max(similarity_scores))
        
        return highest_scores

    def plot_similarity(self, highest_scores, doc_names):
        plt.bar(doc_names, highest_scores)
        plt.title("Highest Cosine Similarity Scores within the Same Category")
        plt.xlabel("Document Name")
        plt.ylabel("Highest Similarity Score")
        plt.ylim(0, 1)
        plt.show()

    def print_highest_scores(self, highest_scores, doc_names):
        for score, name in zip(highest_scores, doc_names):
            print(f"The highest similarity score of {name} within its category is: {score*100:.2f}.")

    def print_topic_words(self, lda_model, feature_names, n_top_words=20):
        for topic_idx, topic in enumerate(lda_model.components_):
            message = f"Topic #{topic_idx}: "
            message += " ".join([feature_names[i]for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)

    def compare_topics(self, new_doc_topics, existing_doc_topics, existing_doc_names):
        for i, new_topic_dist in enumerate(new_doc_topics):
            print(f"Analysis for Document {i+1}:")
            cos_similarities = cosine_similarity([new_topic_dist], existing_doc_topics)
            most_similar_doc_index = np.argmax(cos_similarities)
            most_similar_doc_name = existing_doc_names[most_similar_doc_index]

            print(f"The new document is most similar to {most_similar_doc_name} in terms of topics.")

            for topic_idx, (new_topic_weight, old_topic_weight) in enumerate(zip(new_topic_dist, existing_doc_topics[most_similar_doc_index])):
                if new_topic_weight < old_topic_weight:
                    print(f"Topic {topic_idx}: Could be improved. Current weight: {new_topic_weight :.2f}, Reference weight: {old_topic_weight :.2f}")
                else:
                    print(f"Topic {topic_idx}: Well covered. Current weight: {new_topic_weight :.2f}, Reference weight: {old_topic_weight :.2f+}")

            print("\n")  # New line for readability between documents