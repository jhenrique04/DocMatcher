import spacy

class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_sm')

    def process(self, text):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]