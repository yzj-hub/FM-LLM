import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq
import json

print("load tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained('Mistral-7B-v0.3', 
                                       use_fast=False, 
                                       trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained('Mistral-7B-v0.3',
                                           device_map="auto",
                                           torch_dtype=torch.bfloat16)
print("load model success...")

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            return None
            
        for feature in features:
            for k, v in feature.items():
                if isinstance(v, list):
                    feature[k] = torch.tensor(v)
        
        batch = super().__call__(features)
        return batch

def build_conversation_prompt(example):
    prompt = ""
    
    if example.get('history'):
        for user_msg, assistant_msg in example['history']:
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    
    current_input = example.get('instruction', '')
    if example.get('input'):  
        current_input += example['input']
    
    prompt += f"<|im_start|>user\n{current_input}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"
    
    return prompt.strip()

def process_func(example):
    MAX_LENGTH = 8192  
    
    if not example.get('instruction') or not example.get('output'):
        return None
    
    prompt = build_conversation_prompt(example)
    
    if len(tokenizer(prompt)['input_ids']) > MAX_LENGTH:
        return None
    
    encodings = tokenizer(prompt, 
                         truncation=True,
                         max_length=MAX_LENGTH,
                         padding=False,
                         return_tensors=None)
    
    labels = [-100] * len(encodings['input_ids'])
    
    last_assistant_start = prompt.rindex("<|im_start|>assistant\n")
    assistant_token_start = len(tokenizer(prompt[:last_assistant_start], add_special_tokens=False)['input_ids'])
    
    labels[assistant_token_start:] = encodings['input_ids'][assistant_token_start:]
    
    return {
        "input_ids": encodings['input_ids'],
        "attention_mask": encodings['attention_mask'],
        "labels": labels
    }

print("load dataset...")
dataset = load_dataset('json', data_files='data/train-not-u.json')

def validate_and_process_dataset(dataset):
    valid_examples = []
    for example in dataset['train']:
        processed = process_func(example)
        if processed is not None:
            valid_examples.append(processed)
    return Dataset.from_list(valid_examples)

tokenized_dataset = validate_and_process_dataset(dataset)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

training_args = Seq2SeqTrainingArguments(
    output_dir="output-not-u",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=3e-4,
    warmup_steps=2000,
    max_grad_norm=1.0,
    logging_steps=10,
    num_train_epochs=10,
    save_steps=10,
    save_total_limit=5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    bf16=True,
    fp16=False,
    remove_unused_columns=False,
    optim="adamw_torch",
    dataloader_pin_memory=True,
    group_by_length=True,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_steps=-1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=CustomDataCollator(
        tokenizer,
        padding=True,
        return_tensors="pt"
    ),
)

print("start...")
try:
    trainer.train()
except Exception as e:
    print(f"error : {str(e)}")
finally:
    trainer.save_model()