from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *
import torch

#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = '/home/vqa/00_backup/model-weights/llama3/infp/checkpoint-320'
HUGGINGFACE_TOKEN = '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'

tokenizer = AutoTokenizer.from_pretrained(model_id, token = HUGGINGFACE_TOKEN,)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token = HUGGINGFACE_TOKEN,
    use_cache=True
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=False,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
model.config.pad_token_id = tokenizer.pad_token_id # updating model config
tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
