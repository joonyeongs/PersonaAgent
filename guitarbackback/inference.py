import torch
import json
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel
import csv
import os

HUGGINGFACE_TOKEN = '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'

# Define the path to the directory containing the model
model_path = '/home/vqa/00_backup/model-weights/llama3/infp/'

# Define the model name (optional)
model_name = "checkpoint-320"

score_list = []

question_path = ''



desired_prompt = input()

prompt = f'''<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{desired_prompt} [/INST]
'''

# Specify other parameters
max_seq_length = 512
dtype = torch.bfloat16
load_in_4bit = False
token = HUGGINGFACE_TOKEN

# Load the model and tokenizer

model = torch.load(os.path.join(model_path, model_name))
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name_or_path=model_path,
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=token,
    cache_dir=model_path
)

model.to('cuda:0')

input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=500, num_return_sequences=1, use_cache=True,)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:", generated_text)