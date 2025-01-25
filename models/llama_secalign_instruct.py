#models/llama_secalign_instruct.py

from monitor.models.base import BaseClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

"""
classifier = LlamaClassifier()
result = classifier.check("Test text")
"""



class LlamaClassifier(BaseClassifier):
   def __init__(self):
       model_id = "meta-llama/Llama-3-8b-instruct"
       self.tokenizer = AutoTokenizer.from_pretrained(model_id)
       base_model = AutoModelForCausalLM.from_pretrained(model_id)
       self.model = PeftModel.from_pretrained(base_model, "./adapter_model.safetensors")

   def check(self, text):
       inputs = self.tokenizer(text, return_tensors="pt")
       with torch.no_grad():
           outputs = self.model.generate(**inputs, max_length=200)
       return self.tokenizer.decode(outputs[0])
