import torch
import json
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel
from utils import *
from training_system_prompt import infp_system_prompt, infj_system_prompt, entj_system_prompt, estj_system_prompt
from datasets import Dataset

mbtis = ['infp', 'infj', 'entj', 'estj']
system_prompts = [infp_system_prompt, infj_system_prompt, entj_system_prompt, estj_system_prompt]

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

for mbti, system_prompt in zip(mbtis, system_prompts) :
    input_dir = f"data/generated_data/cleaned_data/{mbti}_pair_data_cleaned.json"
    #input_dir = f"data/generated_data/{mbti}_pair_data_cleaned.json"
    cache_dir = "/home/vqa/model-weights/llama3"
    output_dir = f"/home/vqa/model-weights/llama3/low_lr/{mbti}_cleaned"
    #model_dir = f"/home/vqa/model-weights/llama3/infp_cleaned/models--unsloth--llama-3-8b"

    with open(input_dir, "r", encoding='utf-8') as f:
      dataset = json.load(f)

    HUGGINGFACE_TOKEN = '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'
    max_seq_length = 4096   ##doesn't matter


    model, tokenizer = FastLanguageModel.from_pretrained(
        #model_name = "unsloth/llama-2-7b-chat",
        #model_name = model_dir,
        model_name = "unsloth/llama-3-8b-instruct",
        device_map=device,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16, 
        load_in_4bit = False, 
        token = HUGGINGFACE_TOKEN,
        #cache_dir=cache_dir,
        use_cache=True,
    )

    model.to(device)
    #print('loaded ', model_name)
    EOS_TOKEN = tokenizer.eos_token

    system_prompt = system_prompt

    for i in range(len(dataset)):
      for key in dataset[i].keys():
        if key == 'prompt':
          #dataset[i][key] = f'<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n' + dataset[i][key][0] + '[/INST]' # llama2
          dataset[i][key] = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{dataset[i][key]}<|eot_id|><|start_header_id|>assistant\n' 
        elif key == 'rejected':
          dataset[i][key] = dataset[i][key] + EOS_TOKEN             #'<|eot_id|>'
        else:
          dataset[i][key] = dataset[i][key] + EOS_TOKEN             #'<|eot_id|>'
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
                                      output_dir=output_dir,    ### change the path in respect to your local directory
                                      learning_rate = 1e-6,
                                      weight_decay=0.01,
                                      lr_scheduler_type='cosine',
                                      run_name="infp_llama38B",
                                      fp16 = not torch.cuda.is_bf16_supported(),
                                      bf16 = torch.cuda.is_bf16_supported(),
                                      num_train_epochs=3,
                                      logging_steps=1,
                                      split_batches = False,
                                      use_mps_device = False,
                                      fsdp = False, 
                                      save_strategy="epoch",
                                      #evaluation_strategy="epoch",
                                      #load_best_model_at_end=True,
                                      #save_total_limit=4                                                                  
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
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


'''every dpo training data must follow this format:stat -c %x <directory_path>
  {'prompt': '<|im_start|>system\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.<|im_end|>\n<|im_start|>user\nGenerate an approximately fifteen-word sentence that describes all this data: Midsummer House eatType restaurant; Midsummer House food Chinese; Midsummer House priceRange moderate; Midsummer House customer rating 3 out of 5; Midsummer House near All Bar One<|im_end|>\n<|im_start|>assistant\n',
'chosen': 'Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One.<|im_end|>\n',
'rejected': ' Sure! Here\'s a sentence that describes all the data you provided:\n\n"Midsummer House is a moderately priced Chinese restaurant with a customer rating of 3 out of 5, located near All Bar One, offering a variety of delicious dishes."<|im_end|>\n'}
'''



