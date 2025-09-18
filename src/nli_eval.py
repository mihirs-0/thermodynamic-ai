from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NLI:
    """Natural Language Inference evaluator for detecting contradictions."""
    
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # DeBERTa-MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        self.contradiction_label = 0
    
    @torch.inference_mode()
    def predict(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Returns (label, confidence) where label is one of:
        'contradiction', 'neutral', 'entailment'
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        labels = ['contradiction', 'neutral', 'entailment']
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()
        
        return labels[pred_idx], confidence
    
    def is_contradiction(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        """Returns True if premise and hypothesis contradict with confidence > threshold."""
        label, confidence = self.predict(premise, hypothesis)
        return label == 'contradiction' and confidence > threshold
