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
    get_peft_model
)
import torch
import argparse
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, choices=['sft', 'ppo'], required=True)
args = parser.parse_args()

# Common setup
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset processing functions
def format_instruction(example):
    try:
        conversations = example["conversations"]
        if not isinstance(conversations, list):
            return None
        
        messages = []
        for conv in conversations:
            role = "assistant" if conv.get("from") == "assistant" else "user"
            content = conv.get("value", "").strip()
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
    # Add system message to each conversation
    conversations = []
    for messages in examples["messages"]:
        conv = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
        conv.extend(messages)
        conversations.append(conv)
    
    # Apply chat template
    model_inputs = tokenizer(
        [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations],
        truncation=True,
        padding=True,
        max_length=1024,
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

if args.phase == 'sft':
    print("Starting SFT training phase...")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float16
    )
    
    # Load base model for SFT
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False
    )
    
    # Enable gradient checkpointing
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(base_model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments for SFT
    training_args = TrainingArguments(
        output_dir="./qwen-sft-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
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
    
    # Start training
    print("Starting SFT training...")
    trainer.train()
    
    # Save the trained model
    print("Saving SFT model...")
    trainer.save_model("./qwen-sft-lora-final")

elif args.phase == 'ppo':
    print("Starting PPO training phase...")
    
    # Configure quantization for PPO model
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float16
    )
    
    # Load the SFT model for PPO
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "./qwen-sft-lora-final",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False
    )
    
    # Enable gradient checkpointing if available
    if hasattr(ppo_model, "gradient_checkpointing_enable"):
        ppo_model.gradient_checkpointing_enable()
    
    # PPO config
    ppo_config = PPOConfig(
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        ppo_epochs=2,
        init_kl_coeff=0.05,
        adap_kl_ctrl=True,
        target_kl=1.0,
        gamma=0.99,
        max_grad_norm=0.3,
        optimize_cuda_cache=True
    )
    
    class OpenRLHFRewardPPOTrainer(PPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def compute_reward(self, generated, reference):
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)
            ref_text = tokenizer.decode(reference, skip_special_tokens=True)
            
            content_match = 1.0 if ref_text.lower() in gen_text.lower() else -1.0
            length_ratio = len(gen_text) / len(ref_text)
            length_penalty = min(length_ratio, 1.5) / 1.5
            
            return torch.tensor(content_match * length_penalty)
    
    # Initialize PPO trainer
    ppo_trainer = OpenRLHFRewardPPOTrainer(
        model=ppo_model,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=train_dataset
    )
    
    print("Starting PPO training...")
    for epoch in range(3):
        for batch in ppo_trainer.dataloader:
            query_tensors = tokenizer(
                [ex["prompt"] for ex in batch],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(ppo_trainer.accelerator.device)
            
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            
            rewards = [
                ppo_trainer.compute_reward(res, ref["response"]) 
                for res, ref in zip(response_tensors, batch)
            ]
            
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            print(f"Epoch {epoch} - Stats: {stats}")
    
    print("Saving PPO model...")
    ppo_trainer.save_pretrained("./qwen-rlhf-final")
