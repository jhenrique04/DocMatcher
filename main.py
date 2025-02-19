import os
import numpy as np
import torch
from gensim.models import KeyedVectors
from pdf_reader import PDFReader
from preprocessor import Preprocessor
from bert_model import BertModel
from analyzer import Analyzer
from llm_integration import analyze_with_llm  

def document_to_vec(words, model):
    """Convert document words into an averaged Word2Vec vector."""
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def process_documents(paths, pdf_reader, preprocessor, bert_model, word2vec_model):
    """Extract text, preprocess, analyze, and generate embeddings for documents."""
    analyses, embeddings = [], []
    for path in paths:
        text = pdf_reader.extract_text(path)
        processed_text = preprocessor.process(text)

        # Detailed analysis
        analysis_details = preprocessor.analyze_details(processed_text)
        complexity_score = preprocessor.analyze_complexity(processed_text)
        style_score = preprocessor.analyze_style(processed_text)
        vocabulary_diversity = preprocessor.analyze_vocabulary(processed_text)

        # Extract key-terms
        key_terms = preprocessor.extract_key_terms(processed_text)

        # Calculate topics
        topics = preprocessor.analyze_topics(processed_text)

        # Text Analysis Metrics
        analysis_data = {
            "details": analysis_details,
            "complexity": complexity_score,
            "style": style_score,
            "vocabulary_diversity": vocabulary_diversity,
            "key_terms": key_terms,
            "topics": topics
        }

        # Generate embeddings
        bert_embedding = bert_model.get_embeddings(processed_text)
        w2v_embedding = document_to_vec(processed_text, word2vec_model)
        combined_embedding = torch.from_numpy(np.concatenate(
            (bert_embedding.detach().numpy(), w2v_embedding)))

        embeddings.append(combined_embedding)
        analyses.append((path, text, analysis_data))

    return embeddings, analyses

def compare_with_train(test_embedding, train_embeddings):
    """Compute similarity metrics between test and training embeddings."""
    distances = [torch.dist(test_embedding, train_emb, 2).item() for train_emb in train_embeddings]
    return np.mean(distances)

def main(model_path):
    """Main execution flow for document analysis using Mistral."""
    pdf_reader, preprocessor = PDFReader(), Preprocessor()
    bert_model, analyzer = BertModel('jackaduma/SecBERT'), Analyzer()
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    train_paths = [os.path.join(root, f) for root, _, files in os.walk("train/") for f in files if f.endswith(".pdf")]
    test_paths = [os.path.join(root, f) for root, _, files in os.walk("test/") for f in files if f.endswith(".pdf")]

    train_embeddings, train_analyses = process_documents(train_paths, pdf_reader, preprocessor, bert_model, word2vec_model)
    test_embeddings, test_analyses = process_documents(test_paths, pdf_reader, preprocessor, bert_model, word2vec_model)

    for test_embedding, (test_path, test_text, analysis_data) in zip(test_embeddings, test_analyses):
        print(f"\nAnalyzing document: {test_path}")

        # Compute similarity to training documents
        avg_distance = compare_with_train(test_embedding, train_embeddings)
        similarity_feedback = "The document aligns with training standards." if avg_distance <= 0.21 else "The document differs significantly from training standards."

        # Generate cybersecurity evaluation using Mistral, now passing metrics
        mistral_feedback = analyze_with_llm(test_text, analysis_data)

        # Compile final report
        report = f"""
=== Statistical Analysis ===
{similarity_feedback}

=== Cybersecurity Evaluation (Mistral Response) ===
{mistral_feedback}
"""

        print(report)

        # Save the report to a file
        with open(f"report_{os.path.basename(test_path)}.txt", "w", encoding="utf-8") as f:
            f.write(report)

if __name__ == "__main__":
    main('word2Vec_models/skip_s50.txt')
