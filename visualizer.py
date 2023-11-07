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
    plt.show()

  def print_highest_scores(self, highest_scores, doc_names):
    for score, name in zip(highest_scores, doc_names):
      print(f"The highest similarity score of {name} within its category is: {score*100:.2f}.")