# rewrite_prompts.py


from llama2 import LlamaClassifier
import sys

def main():
   if len(sys.argv) != 3:
       print("Usage: python rewrite_prompts.py input_file output_file")
       sys.exit(1)
       
   input_file, output_file = sys.argv[1:3]
   classifier = LlamaClassifier()
   
   with open(input_file, 'r') as f:
       prompts = [line.strip() for line in f if line.strip()]
   
   rewritten = classifier.rewrite_batch(prompts)
   
   with open(output_file, 'w') as f:
       for orig, new in zip(prompts, rewritten):
           f.write(f"Original: {orig}\nRewritten: {new}\n\n")

if __name__ == "__main__":
   main()
