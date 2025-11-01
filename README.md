### FM-LLM

FM-LLM is an approach designed to automatically build an SPL feature model from textual software requirements by augmenting LLMs with use case relationships. The crux of this approach is to leverage use case relationships to product an assistive chain of thought to LLMs, aiming to address challenges posed by complex reasoning when extracting feature constraints.

### Environment Setup

Open your terminal and use the following command to install packages:

```bash
pip install -r requirements.txt
```

### Datasets

The datasets are located in the `fine-tuning/data` folder. The `train-has-u.json` contains datasets enhanced with use case relationships, while the `train-not-u.json` stores datasets without use case relationship enhancement. The `test.json` is the test dataset.

### Fine-Tuning

This experiment use the following three models, you can download it and save it to your local folder.

| model          | url                                                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepSeek-R1-7B | [https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| Mistral 7B     | [https://www.modelscope.cn/models/LLM-Research/Mistral-7B-Instruct-v0.3](https://www.modelscope.cn/models/LLM-Research/Mistral-7B-Instruct-v0.3)     |
| LLaMA3-8B      | [https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct) |

Then open file fine-tuning/lora-has-u.py and file fine-tuning/lora-not-u.py, change the model path in the code below to your local model path.

```python
tokenizer = AutoTokenizer.from_pretrained('Mistral-7B-v0.3', 
                                       use_fast=False, 
                                       trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained('Mistral-7B-v0.3',
                                           device_map="auto",
                                           torch_dtype=torch.bfloat16)
```

Execute command:

```
cd finie-tunning
python lora-has-u.py
```

or

```
cd finie-tunning
python lora-not-u.py
```

### Predict

Then open file fine-tuning/predict.py, change the model path in the code below to your local model path.

```python
model_path = 'Mistral-7B-v0.3'
lora_path = '../lora-model/Mistral-has-u'
```

Execute command:

```
cd finie-tunning
python predict.py
```

### Note:

The LoRA-model file contains the fine-tuned model from this experiment.
