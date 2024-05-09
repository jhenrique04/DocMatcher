from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class BertModel:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path)
        self.model.eval()

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt',
                                padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs[0].mean(dim=1)
