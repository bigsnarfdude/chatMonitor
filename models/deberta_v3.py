# monitor/models/deberta.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from .base import BaseClassifier

class InjectionClassifier(BaseClassifier):
    def __init__(self, model_name="ProtectAI/deberta-v3-base-prompt-injection-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
            device=self.device
        )

    def check(self, text, threshold=0.8):
        result = self.classifier(text)[0]
        return {
            'is_injection': result['label'] == 'INJECTION' and result['score'] > threshold,
            'score': result['score'],
            'label': result['label']
        }
