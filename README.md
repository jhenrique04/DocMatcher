# DocMatcher

## Overview - WIP

This project is focused on analyzing documents related to cybersecurity risks and comparing them against a dataset of expert-reviewed documents, considered as the standard for each document type. The primary objective is to provide feedback on the analyzed documents based on the established expert-reviewed standards.

## How to run

If you want to use a virtual enviroment:

    source path/to/venv/bin/activate  # Linux/macOS
    path\to\venv\Scripts\activate      # Windows

Install Python dependencies:

    pip install -r requirements.txt

Create "word2Vec_models", "train" and "test" folders

    mkdir word2Vec_models train test

Place your preferred Word2Vec model in the word2Vec_models folder and update the path in the 'main' function call. Distribute the files you want to analyze into the train and test folders, ideally using 80% of the data for training and 20% for testing.

Install a spaCy language core (this project uses 'pt_core_news_lg'):

    python -m spacy download your_spacy_core

You may also need to download the vader_lexicon from nltk, heres how you can do that in a Python shell:

    import nlkt
    nltk.download("vader_lexicon")

Then change the nltk data path under preprocessor.py and should be good to go at this point to run the main.ipynb file.

     nltk.data.path.append('your_ntlk_data_path')

## Project Structure

The project directory is organized as follows:

- `pdf_reader/`: Contains the code for PDF text extraction.
- `preprocessor/`: Includes preprocessing functions for document analysis.
- `bert_model/`: Contains the fine-tuned BERT model for text embeddings.
- `analyzer/`: Code for document analysis and feedback generation.
- `word2vec_models/`: Word2Vec models for word embeddings.
- `train/`: Store your training documents.
- `test/`: Store the documents you want to analyze and generate feedback for.

The last two directories could not be uploaded on Github for copyright reasons.

## The project is still under development, feel free to contribute.
