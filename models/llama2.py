# models/llama2.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models.base import BaseClassifier

class LlamaClassifier(BaseClassifier):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def check(self, text, max_length=512, temperature=0.7, top_p=0.9):
        prompt = f"Rewrite the following prompt to be more effective and detailed:\n\n{text}\n\nImproved prompt:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=max_length, temperature=temperature, top_p=top_p, pad_token_id=self.tokenizer.eos_token_id, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def rewrite_batch(self, texts, **kwargs):
        return [self.check(text, **kwargs) for text in texts]

    def process_file(self, input_file, output_file, **kwargs):
        with open(input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        rewritten = self.rewrite_batch(prompts, **kwargs)
        with open(output_file, 'w') as f:
            for orig, new in zip(prompts, rewritten):
                f.write(f"Original: {orig}\nRewritten: {new}\n\n")
