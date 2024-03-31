import torch
import json
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel


HUGGINGFACE_TOKEN = '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'
max_seq_length = 2048   ##doesn't matter

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-chat",
    max_seq_length = max_seq_length,
    dtype = torch.float16, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False, # Use 4bit quantization to reduce memory usage. Can be False.
    token = HUGGINGFACE_TOKEN,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)

training_args = TrainingArguments(per_device_train_batch_size=1,
                                  output_dir="/content/drive/MyDrive/mbti",    ### change the path in respect to your local directory
                                  lr_scheduler_type='cosine',
                                  run_name="mbti_llama2_chat_7B",
                                  )
model_ref = None
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=data,
    tokenizer=tokenizer,
)
dpo_trainer.train()



data_path  = ''                                 ####your_data_path_here
with open(data_path, "r", encoding='utf-8') as f:
  data = json.load(f)


