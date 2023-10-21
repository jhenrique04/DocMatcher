from preprocessor import Preprocessor
from pdf_reader import PDFReader
from vectorizer import Vectorizer
import numpy as np

# Initialize classes
preprocessor = Preprocessor()
pdf_reader = PDFReader()
vectorizer = Vectorizer("cbow_s1000.txt")

# Paths to the template documents
good_docs_paths = [""]  # 100-90%
great_docs_paths = [""]  # 75%
avg_docs_paths = [""]  # 50%
below_avg_docs_paths = [""]  # 25%
bad_docs_paths = [""]  # <25%

# Extract and preprocess the template documents
good_docs = [preprocessor.process(pdf_reader.extract_text(file)) for file in good_docs_paths]
great_docs = [preprocessor.process(pdf_reader.extract_text(file)) for file in great_docs_paths]
avg_docs = [preprocessor.process(pdf_reader.extract_text(file)) for file in avg_docs_paths]
below_avg_docs = [preprocessor.process(pdf_reader.extract_text(file)) for file in below_avg_docs_paths]
bad_docs = [preprocessor.process(pdf_reader.extract_text(file)) for file in bad_docs_paths]

all_docs = good_docs + great_docs + avg_docs + below_avg_docs + bad_docs

# Vectorize all templates
all_docs_vectors = [vectorizer.calculate_document_vector(text) for text in all_docs]

# Process a new document and calculate similarity
doc_path = ""
doc = preprocessor.process(pdf_reader.extract_text(doc_path))
doc_vector = vectorizer.calculate_document_vector(doc)
similarity_scores = vectorizer.calculate_similarity(doc_vector, all_docs_vectors)

# Calculate final score
score = (
    np.mean(similarity_scores[:len(good_docs)]) +
    0.75 * np.mean(similarity_scores[len(good_docs):len(good_docs)+len(great_docs)]) +
    0.5 * np.mean(similarity_scores[len(good_docs)+len(great_docs):len(good_docs)+len(great_docs)+len(avg_docs)]) +
    0.25 * np.mean(similarity_scores[len(good_docs)+len(great_docs)+len(avg_docs):len(good_docs)+len(great_docs)+len(avg_docs)+len(below_avg_docs)]) +
    0.0 * np.mean(similarity_scores[-len(bad_docs):])
)

print(f"Score: {score:.2f}")