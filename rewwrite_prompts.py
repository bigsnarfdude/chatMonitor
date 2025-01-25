# rewrite_prompts.py

from monitor.models.llama2 import LlamaClassifier

classifier = LlamaClassifier()
classifier.process_file('prompts.txt', 'rewritten_prompts.txt')
