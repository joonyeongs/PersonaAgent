import torch
import json
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel


with open("data/generated_data/infp_total_data.json", "r", encoding='utf-8') as f:
  dataset = json.load(f)

HUGGINGFACE_TOKEN = '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'
max_seq_length = 4096   ##doesn't matter

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-chat",
    max_seq_length = max_seq_length,
    dtype = torch.float16, 
    load_in_4bit = False, 
    token = HUGGINGFACE_TOKEN,
    cache_dir="/workspace/model-weights"
)

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

model = model.to("cuda:4")

training_args = TrainingArguments(per_device_train_batch_size=8,
                                  gradient_accumulation_steps = 16,
                                  output_dir="outputs",    ### change the path in respect to your local directory
                                  lr_scheduler_type='cosine',
                                  run_name="infp_llama2_chat_7B",
                                  num_train_epochs=1,
                                  logging_steps=1,
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



