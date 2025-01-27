from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
import torch
import argparse

# Load the tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset processing functions with thinking tags preservation
def format_instruction(example):
    try:
        conversations = example["conversations"]
        if not isinstance(conversations, list):
            return None
        
        messages = []
        for conv in conversations:
            role = "assistant" if conv.get("from") == "assistant" else "user"
            # Remove strip() to preserve tags
            content = conv.get("value", "")
            if content:
                messages.append({"role": role, "content": content})
                
        if len(messages) < 2:
            return None
            
        return {
            "messages": messages
        }
    except Exception as e:
        print(f"Error in format_instruction: {str(e)}")
        return None

def prepare_train_features(examples):
    # Update system message to include thinking process instructions
    system_message = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>"""
    
    conversations = []
    for messages in examples["messages"]:
        conv = [{"role": "system", "content": system_message}]
        conv.extend(messages)
        conversations.append(conv)
    
    # Apply chat template
    model_inputs = tokenizer(
        [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations],
        truncation=True,
        padding=True,
        max_length=2048,  # Increased for longer context
        return_tensors=None
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Load and process dataset
print("Loading dataset...")
dataset = load_dataset("bespokelabs/Bespoke-Stratos-35k", split="train")
print(f"Initial dataset size: {len(dataset)}")

print("Processing dataset...")
formatted_dataset = dataset.map(format_instruction)
formatted_dataset = formatted_dataset.filter(lambda x: x is not None and x.get("messages"))

train_dataset = formatted_dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=formatted_dataset.column_names,
    batch_size=100
)

print(f"Final dataset size: {len(train_dataset)}")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16
)

# Load the model from epoch 1 checkpoint
print("Loading model from epoch 1 checkpoint...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=False
)

# Load the trained LoRA adapter from epoch 1
model = PeftModel.from_pretrained(
    base_model,
    "./qwen-sft-lora-final",  # Path to epoch 1 checkpoint
    is_trainable=True
)

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# Training arguments for epoch 2
training_args = TrainingArguments(
    output_dir="./qwen-sft-lora-epoch2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,  # Reduced learning rate for epoch 2
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    max_grad_norm=0.3,
    warmup_ratio=0.01,  # Reduced warmup for epoch 2
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    ddp_find_unused_parameters=False,
    report_to="wandb"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        padding=True
    )
)

# Start training epoch 2
print("Starting epoch 2 training...")
trainer.train()

# Save the trained model
print("Saving epoch 2 model...")
trainer.save_model("./qwen-sft-lora-epoch2-final")
