import spacy
import nltk
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer

class Preprocessor:
    def __init__(self, model='pt_core_news_lg'):
        self.nlp = spacy.load(model)
        nltk.data.path.append('/home/jhenrique/myenv/nltk_data')
        self.sia = SentimentIntensityAnalyzer()

    def process(self, text):
        text = self._clean_text(text)
        doc = self.nlp(text)
        processed_text = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(processed_text)

    def _clean_text(self, text):
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\t", " ")

        text = " ".join(text.split())

        return text

    def analyze_details(self, text):
        doc = self.nlp(text)
        num_sentences = len(list(doc.sents))
        sentiment = self.sia.polarity_scores(text)
        return {
            "num_sentences": num_sentences,
            "sentiment": sentiment
        }

    def analyze_complexity(self, text):
        readability_score = textstat.flesch_kincaid_grade(text)
        return readability_score

    def analyze_style(self, text):
        doc = self.nlp(text)
        passive_sentences = sum(1 for sent in doc.sents if self.is_passive(sent))
        return passive_sentences

    def is_passive(self, sentence):
        passive = False
        for token in sentence:
            if token.dep_ == 'auxpass' or (token.dep_ == 'aux' and token.head.dep_ == 'acl'):
                passive = True
                break
        return passive

    def analyze_vocabulary(self, text):
        doc = self.nlp(text)
        unique_words = set(token.text.lower() for token in doc if token.is_alpha)
        
        if len(doc) == 0:
            return 0.0
        
        lexical_diversity = len(unique_words) / len(doc)
        return lexical_diversity

    def extract_key_terms(self, text):
        key_terms_list = ['segurança', 'risco', 'vulnerabilidade', 'ataque', 'criptografia', 'firewall', 'cyber']
        
        doc = self.nlp(text)
        document_words = set([token.text.lower() for token in doc])

        # Find key terms that are in the document
        key_terms = set(term for term in key_terms_list if term in document_words)
        return key_terms

    def analyze_topics(self, text):
        doc = self.nlp(text)
        word_freq = {}
        for token in doc:
            if token.is_alpha and not token.is_stop:
                word_freq[token.lemma_] = word_freq.get(token.lemma_, 0) + 1

        # Sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True)
        topics = [word for word, frequency in sorted_words[:5]]
        return topics
