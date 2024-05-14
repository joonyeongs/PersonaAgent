from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


'''
This Code runs only for single-turn. If you want to have multi-turn conversations with the model, use llama2_inference.py
'''

def llama2(system, desired_prompt):
    # Set the path to your local model directory
    model_directory = '/home/vqa/model-weights/llama2/infp_cleaned/checkpoint-400'


    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_directory, use_cache=True)
    model = AutoModelForCausalLM.from_pretrained(model_directory, use_cache=True)

    model.to('cuda:0')
    #system_prompt = 'You have a INFP personality'
    system_prompt = system    
    prompt = f'<s>[INST] <<SYS>>\n You are an INFP\n<</SYS>>\n{desired_prompt} [/INST]\nMy score to the answer between 1 and 7 is' #prompt format for llama2

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

    # Generate a response
    output = model.generate(input_ids, max_length=500)
    print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))

    return tokenizer.decode(output[0][input_ids.shape[-1]:])




def llama3(system, desired_prompt, messages):
    model_directory = '/home/vqa/model-weights/llama3/infp/checkpoint-400'

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        
    )
    EOS_TOKEN = tokenizer.eos_token
    
    if messages[0]["role"] != "system":
         messages.append({"role": "system", "content": system})   
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
    #print(real_response)
    messages.append({"role": "assistant", "content": real_response})
    return real_response
