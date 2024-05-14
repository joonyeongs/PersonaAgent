from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def llama3_response(desired_prompt):
    model_directory = '/home/vqa/00_backup/model-weights/llama3/infp/checkpoint-400'

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        
    )
    EOS_TOKEN = tokenizer.eos_token

    messages = []
    messages.append({"role": "system", "content": "You have a INFP personality"})    
    messages.append({"role": "user", "content": desired_prompt})        
    

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_length=128, 
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        use_cache=True
    )
    response = outputs[0][input_ids.shape[-1]:]
    real_response = tokenizer.decode(response, skip_special_tokens=False)
    print(real_response)
    return real_response