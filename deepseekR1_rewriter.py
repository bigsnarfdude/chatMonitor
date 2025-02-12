# Install necessary libraries
# pip install -qU transformers accelerate

# Import required modules
from transformers import AutoTokenizer, pipeline
import torch

# Model configuration
model_name = "NaniDAO/deepseek-r1-qwen-2.5-32B-ablated"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline(
    task="text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# File paths
input_file = "prompts.txt"  # Input file containing prompts (one per line)
output_file = "rewritten.txt"  # Output file for rewritten prompts

# Process prompts from the input file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        prompt_content = line.strip()  # Remove whitespace
        if prompt_content:  # Skip empty lines
            # Create the structured message for the model
            messages = [{"role": "user", "content": prompt_content}]
            
            # Generate the prompt using the tokenizer's chat template
            prompt = tokenizer.apply_chat_template(
                conversation=messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate rewritten prompt
            outputs = text_generator(
                prompt, 
                max_new_tokens=256, 
                do_sample=True, 
                temperature=0.7, 
                top_k=50, 
                top_p=0.95
            )
            
            # Extract and save the generated text
            rewritten_prompt = outputs[0]["generated_text"].strip()
            outfile.write(rewritten_prompt + "\n")
            print(f"Processed: {prompt_content}")  # Optional progress log
