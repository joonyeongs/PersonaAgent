import torch
import json
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel
from utils import *
from training_system_prompt import infp_system_prompt, infj_system_prompt, entj_system_prompt, estj_system_prompt
from datasets import Dataset
import argparse
import copy
import os

mbtis = ['infp', 'infj', 'entj', 'estj']
system_prompts = [infp_system_prompt, infj_system_prompt, entj_system_prompt, estj_system_prompt]
reverse_traits = ["T", "E", "J", "S"]
machine_path = ['data/machine_mindset/'+path for path in os.listdir('data/machine_mindset')]


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def switch_chosen_rejected(dataset):
   for data in dataset:
      temp = copy.deepcopy(data['rejected'])
      data['rejected'] = data['chosen']
      data['chosen'] = temp

   return dataset


def get_machine_dataset(traits):
   path_list = []
   dataset = []
   for trait in traits:
      if trait.upper() in ['I', 'E']:
         path_list.append(machine_path[1])

      if trait.upper() in ['F', 'T']:
         path_list.append(machine_path[0])

      if trait.upper() in ['P', 'J']:
         path_list.append(machine_path[2])

      if trait.upper() in ['N', 'S']:
         path_list.append(machine_path[3])
         
   for path, trait in zip(path_list, traits):
      with open(path, 'r', encoding='utf-8') as f:
         data = json.load(f)
      temp = []
      for i in range(len(data)):
        if "rejected" in data[i].keys():
            temp.append(data[i])
      data = temp
      if trait in reverse_traits:
          data = switch_chosen_rejected(data)
      dataset += data[:1000]
   return dataset        
   




def main(args):
  device = get_device_map()
  for mbti, system_prompt in zip(mbtis, system_prompts) :
      input_dir = args.dataset_dir
      #input_dir = f"data/generated_data/{mbti}_pair_data_cleaned.json"
      #cache_dir = "/home/vqa/model-weights/llama3"
      output_dir = args.output_dir
      model_dir = args.model_path
      save_path = args.save_path
      trait = args.mbti_trait
      print('\n######################################')
      print(f'saving at {output_dir} with traits {trait} changed')
      print('\nd######################################')
      if trait is not None:
         dataset = get_machine_dataset(trait)
      
      else:                                               #### for normal training
        with open(input_dir, "r", encoding='utf-8') as f:
          dataset = json.load(f)
        
        

      HUGGINGFACE_TOKEN = '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'
      max_seq_length = 4096   ##doesn't matter


      model, tokenizer = FastLanguageModel.from_pretrained(
          model_name = model_dir,
          device_map=device,
          max_seq_length = max_seq_length,
          dtype = torch.bfloat16, 
          load_in_4bit = False, 
          token = HUGGINGFACE_TOKEN,
          #cache_dir=cache_dir,
          use_cache=True,
      )

      model.to(device)
      EOS_TOKEN = tokenizer.eos_token

      system_prompt = system_prompt
      temp = []
      for i in range(len(dataset)):
        if "rejected" in dataset[i].keys():
            temp.append(dataset[i])
      dataset = temp
              

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
                    
      dataset = Dataset.from_dict({key: [d[key] for d in dataset] for key in dataset[0]})
      #print(dataset['prompt'])    ### maps that dataset to 'datasets.arrow_dataset.Dataset' object

      if args.model_path == 'unsloth/llama-3-8b-instruct':
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
      model.to(device)

      #tokenizer = tokenizer.to("cuda:5")

      training_args = TrainingArguments(per_device_train_batch_size=8,
                                        gradient_accumulation_steps = 16,
                                        output_dir=output_dir,    ### change the path in respect to your local directory
                                        learning_rate = 1e-5,
                                        weight_decay=0.01,
                                        lr_scheduler_type='cosine',
                                        #run_name="infp_llama38B",
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
      print(f'saving at {output_dir}')
      dpo_trainer.save_model(output_dir)
      tokenizer.save_pretrained(output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="unsloth--llama-3-8b-instruct", required=True)
    parser.add_argument('--dataset_dir', type=str, default='', required=False)
    parser.add_argument('--save_path', type=str, default="/home/vqa/model-weights/llama3", required=False)
    parser.add_argument('--output_dir', type=str, default="/home/vqa/data/outputs/mbti/official_metric", required=True)
    parser.add_argument('--mbti_trait', type=list, default=None, required=False)    ##바꾸고싶은애
    args = parser.parse_args()

main(args)







'''every dpo training data must follow this format:stat -c %x <directory_path>
  {'prompt': '<|im_start|>system\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.<|im_end|>\n<|im_start|>user\nGenerate an approximately fifteen-word sentence that describes all this data: Midsummer House eatType restaurant; Midsummer House food Chinese; Midsummer House priceRange moderate; Midsummer House customer rating 3 out of 5; Midsummer House near All Bar One<|im_end|>\n<|im_start|>assistant\n',
'chosen': 'Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One.<|im_end|>\n',
'rejected': ' Sure! Here\'s a sentence that describes all the data you provided:\n\n"Midsummer House is a moderately priced Chinese restaurant with a customer rating of 3 out of 5, located near All Bar One, offering a variety of delicious dishes."<|im_end|>\n'}
'''





