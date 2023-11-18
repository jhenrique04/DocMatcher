from sklearn.preprocessing import StandardScaler
import numpy as np

class Extractor:
  def __init__(self, vectorizer, pdf_reader, preprocessor):
    self.vectorizer = vectorizer
    self.pdf_reader = pdf_reader
    self.preprocessor = preprocessor
    self.scaler = StandardScaler()

  def extract_features(self, file_paths):
    word2vec_features = []
    doc_lengths = []

    for file_path in file_paths:
      text = self.pdf_reader.extract_text(file_path)
      processed_text = self.preprocessor.process(text)
      vectorized_text = self.vectorizer.calculate_document_vector(processed_text)
      word2vec_features.append(vectorized_text)

      doc_length = self.pdf_reader.doc_lenght(file_path)
      doc_lengths.append(doc_length)

    doc_lengths_scaled = self.scaler.fit_transform(np.array(doc_lengths).reshape(-1, 1))
    combined_features = np.hstack((word2vec_features, doc_lengths_scaled))

    return combined_features