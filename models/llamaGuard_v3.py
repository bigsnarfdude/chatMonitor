from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base import BaseClassifier

"""
classifier = LlamaGuardClassifier()
result = classifier.check("your text")
"""

class LlamaGuardClassifier(BaseClassifier):
    def __init__(self, model_id="meta-llama/Llama-Guard-3-8B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
    
    def check(self, text: str) -> dict:
        chat = [{"role": "user", "content": text}]
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt"
        ).to(self.device)
        
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            pad_token_id=0
        )
        response = self.tokenizer.decode(
            output[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        return {
            'is_unsafe': "unsafe" in response.lower(),
            'response': response
        }
