import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class DocumentClustering:
    
    def __init__(self, pdf_reader, preprocessor):
        self.pdf_reader = pdf_reader
        self.preprocessor = preprocessor

    def extract_and_preprocess(self, file_paths):
        processed_files = []
        for file in tqdm(file_paths, desc="Processing files", leave=True, bar_format='{l_bar}{bar}|'):
            processed_file = self.preprocessor.process(self.pdf_reader.extract_text(file))
            processed_files.append(processed_file)
        return processed_files

    def visualize_clusters_distribution(self, labels, k):
        plt.hist(labels, bins=np.arange(0, k+1) - 0.5, rwidth=0.5, align='mid')
        plt.xticks(range(k))
        plt.xlabel('Cluster')
        plt.ylabel('Número de Documentos')
        plt.title('Distribuição dos Documentos nos Clusters')
        plt.show()

    def visualize_distances_to_centroid(self, doc_paths, distances_to_centroid):
        plt.bar(range(len(doc_paths)), distances_to_centroid)
        plt.xlabel('Documentos')
        plt.xticks(range(len(doc_paths)), doc_paths, rotation=90)
        plt.ylabel('Distância até o Centróide')
        plt.title('Distância dos Documentos até o Centróide de seus Clusters')
        plt.tight_layout()
        plt.show()