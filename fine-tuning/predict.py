from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from peft import PeftModel

model_path = 'Mistral-7B-v0.3'
lora_path = '../lora-model/Mistral-has-u'

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建生成配置
generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    generation_config=generation_config
)
model = PeftModel.from_pretrained(model, lora_path)

def chat(prompt):
    try:
        # 初始化对话历史 - 移除了系统提示词
        messages = []
        messages.append({"role": "user", "content": prompt})
        
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        model_inputs = {
            'input_ids': inputs.input_ids.to('cuda'),
            'attention_mask': inputs.attention_mask.to('cuda')
        }
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8192,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        
        response = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # messages.append({"role": "assistant", "content": response})
        return response
        
    finally:
        del model_inputs, generated_ids
        torch.cuda.empty_cache()

# 交互式对话循环
print("Start the conversation. Enter ‘quit’ to end the conversation.")
while True:
    user_input = input("\nuser: ")
    if user_input.lower() == 'quit':
        print("end")
        break
        
    response = chat(user_input)
    print("\nAI: " + response)