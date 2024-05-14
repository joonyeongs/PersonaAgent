import torch
import json
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel
from utils import *

from datasets import Dataset


with open("data/generated_data/infp_total_data.json", "r", encoding='utf-8') as f:
  dataset = json.load(f)

HUGGINGFACE_TOKEN = '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'
max_seq_length = 4096   ##doesn't matter


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-chat",
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16, 
    load_in_4bit = False, 
    token = HUGGINGFACE_TOKEN,
    cache_dir="/workspace/model-weights"
)

EOS_TOKEN = tokenizer.eos_token

for i in range(len(dataset)):
  for key in dataset[i].keys():
    dataset[i][key] = dataset[i][key][0] + EOS_TOKEN
    #print('this type', type(dataset[i][key])) 
dataset = Dataset.from_dict({key: [d[key] for d in dataset] for key in dataset[0]})    ### maps that dataset to 'datasets.arrow_dataset.Dataset' object
    

#model = model.to("cuda:5")
model = FastLanguageModel.get_peft_model(
    model=model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)
model = model.to("cuda:0")

#tokenizer = tokenizer.to("cuda:5")

training_args = TrainingArguments(per_device_train_batch_size=8,
                                  gradient_accumulation_steps = 16,
                                  output_dir="/workspace/model-weights",    ### change the path in respect to your local directory
                                  lr_scheduler_type='linear',
                                  run_name="infp_llama2_chat_7B",
                                  fp16 = not torch.cuda.is_bf16_supported(),
                                  bf16 = torch.cuda.is_bf16_supported(),
                                  num_train_epochs=4,
                                  logging_steps=1,
                                  split_batches = False,
                                  use_mps_device = False,
                                  fsdp = False, 
                                  save_strategy="epoch",
                                  #evaluation_strategy="epoch",
                                  #load_best_model_at_end=True,
                                  save_total_limit=3                                                                    
                                  )

dpo_trainer = DPOTrainer(
    model=model,
    ref_model = None,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_length=max_seq_length,
)

dpo_trainer.train()
dpo_trainer.save_model("/workspace/model-weights")
tokenizer.save_pretrained("/workspace/model-weights")



