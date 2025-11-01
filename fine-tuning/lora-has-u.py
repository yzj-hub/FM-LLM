import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq
import json
import os

print("load tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    'Mistral-7B-v0.3', 
    use_fast=False, 
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    'Mistral-7B-v0.3',
    device_map="auto",
    torch_dtype=model_dtype
)
print(f"load model success，dtype: {model_dtype}")


class DensityRatioEstimator(torch.nn.Module):
    def __init__(self, hidden_size=4096, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size // 4, dtype=dtype)
        self.dense2 = torch.nn.Linear(hidden_size // 4, 1, dtype=dtype)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = x.to(self.dtype)
        x1 = self.activation(self.dense1(x))
        x2 = self.dropout(x1)
        return self.dense2(x2).squeeze(-1)


class MutualInformationLossWrapper:
    def __init__(self, model, tokenizer, alpha=0.5, lambda_=1.0, device="cuda"):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.device = device
        self.tokenizer = tokenizer
        self.dtype = model.dtype
        
        hidden_size = model.config.hidden_size
        self.density_ratio_estimator = DensityRatioEstimator(
            hidden_size=hidden_size,
            dtype=self.dtype
        ).to(device)
        
        self.estimator_optimizer = torch.optim.AdamW(
            self.density_ratio_estimator.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

    def _compute_variational_mi(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        with torch.no_grad():  
            hidden_detached = hidden_states.detach().clone().to(self.dtype)

            d_pos = self.density_ratio_estimator(hidden_detached)
            first_term = d_pos.mean()

            shuffled_indices = torch.randperm(batch_size).to(self.device)
            hidden_shuffled = hidden_detached[shuffled_indices]
            d_neg = self.density_ratio_estimator(hidden_shuffled)
            
            exp_d_neg = torch.exp(d_neg.clamp(max=10))
            second_term = torch.log(exp_d_neg.mean() + 1e-8)

            mi_lower_bound = first_term - second_term

        self.estimator_optimizer.zero_grad()
        d_pos_new = self.density_ratio_estimator(hidden_detached.clone())
        d_neg_new = self.density_ratio_estimator(hidden_shuffled.clone())
        first_term_new = d_pos_new.mean()
        exp_d_neg_new = torch.exp(d_neg_new.clamp(max=10))
        second_term_new = torch.log(exp_d_neg_new.mean() + 1e-8)
        mi_lower_bound_new = first_term_new - second_term_new

        (-mi_lower_bound_new).backward()
        self.estimator_optimizer.step()

        return mi_lower_bound.to(self.dtype)

    def compute_total_loss(self, model_outputs, labels):
        logits = model_outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(self.device)
        
        active_loss = shift_labels != -100
        active_logits = shift_logits[active_loss].to(self.dtype)
        active_labels = shift_labels[active_loss]
        
        lm_loss = F.cross_entropy(
            active_logits.view(-1, active_logits.size(-1)),
            active_labels.view(-1)
        ).to(self.dtype)

        hidden_states = model_outputs.hidden_states[-1]
        mi_estimate = self._compute_variational_mi(hidden_states)  
        splmi_loss = (-self.lambda_ * mi_estimate).to(self.dtype)

        total_loss = (self.alpha * lm_loss + (1 - self.alpha) * splmi_loss).to(self.dtype)

        model_outputs.lm_loss = float(lm_loss.item())
        model_outputs.mi_loss = float(splmi_loss.item())
        model_outputs.mi_estimate = float(mi_estimate.item())

        return total_loss


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
    
    encodings = tokenizer(
        prompt, 
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )
    
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
dataset = load_dataset('json', data_files='data/train-has-u.json')

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
model.print_trainable_parameters()


class MutualInfoSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, mi_alpha=0.5, mi_lambda=1.0, detect_anomaly=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mi_loss_wrapper = MutualInformationLossWrapper(
            model=self.model,
            tokenizer=self.tokenizer,
            alpha=mi_alpha,
            lambda_=mi_lambda,
            device=self.args.device
        )
        self.detect_anomaly = detect_anomaly

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        with torch.autograd.set_detect_anomaly(self.detect_anomaly):
            labels = inputs.pop("labels")
            inputs["output_hidden_states"] = True
            outputs = model(**inputs)

            total_loss = self.mi_loss_wrapper.compute_total_loss(outputs, labels)

            logs = {
                "lm_loss": outputs.lm_loss,
                "mi_loss": outputs.mi_loss,
                "mi_estimate": outputs.mi_estimate
            }
            self.log(logs)

            return (total_loss, outputs) if return_outputs else total_loss


training_args = Seq2SeqTrainingArguments(
    output_dir="output-has-u",
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
    logging_strategy="steps",
    report_to="none"
)

trainer = MutualInfoSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=CustomDataCollator(
        tokenizer,
        padding=True,
        return_tensors="pt"
    ),
    tokenizer=tokenizer,
    mi_alpha=0.5,
    mi_lambda=1.0,
    detect_anomaly=False
)

print("start...")
try:
    trainer.train()
except Exception as e:
    print(f"error : {str(e)}")
finally:
    print("save model...")
    trainer.save_model()
    torch.save(
        {
            "state_dict": trainer.mi_loss_wrapper.density_ratio_estimator.state_dict(),
            "dtype": str(trainer.mi_loss_wrapper.dtype)
        },
        f"{training_args.output_dir}/density_ratio_estimator.pth"
    )
    tokenizer.save_pretrained(training_args.output_dir)
