import ollama
import time
from pathlib import Path

def rewrite_prompts(input_file="prompts.txt", output_file="rewritten.txt", model="tarruda/neuraldaredevil-8b-abliterated:fp16"):
    try:
        # Read input prompts
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts = f.read().strip().split('\n')
        
        
        rewritten_prompts = []
        
        # Process each prompt
        for prompt in prompts:
            try:
                # Create the instruction for rewriting
                instruction = f"""Rewrite prompt using different wording:

Original: {prompt}

Rewritten:"""
                
                # Get response from Ollama
                response = ollama.chat(model=model, messages=[
                    {
                        'role': 'user',
                        'content': instruction
                    }
                ])
                
                # Extract the rewritten prompt
                rewritten = response['message']['content'].strip()
                rewritten_prompts.append(rewritten)
                
                # Add a small delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing prompt: '{prompt}'\nError: {str(e)}")
                rewritten_prompts.append(f"# Error processing: {prompt}")
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rewritten_prompts))
            
        print(f"Successfully processed {len(rewritten_prompts)} prompts")
        print(f"Output written to: {output_file}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        model_name = "tarruda/neuraldaredevil-8b-abliterated:fp16"
        print(f"Using model: {model_name}")
        rewrite_prompts(model=model_name)
        
    except Exception as e:
        print(f"Error: {str(e)}")
