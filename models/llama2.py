from transformers import AutoModelForCausalLM, AutoTokenizer


'''
This Code runs only for single-turn. If you want to have multi-turn conversations with the model, use llama2_inference.py
'''

def llama2_response(desired_prompt):
    # Set the path to your local model directory
    model_directory = '/home/vqa/00_backup/model-weights/llama2/infp/checkpoint-400'


    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_directory, use_cache=True)
    model = AutoModelForCausalLM.from_pretrained(model_directory, use_cache=True)

    model.to('cuda:0')
    system_prompt = 'You have a INFP personality'        
    prompt = f'<s>[INST] <<SYS>>\n You are an INFP\n<</SYS>>\n{desired_prompt} [/INST]' #prompt format for llama2

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

    # Generate a response
    output = model.generate(input_ids, max_length=500)
    print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))

    return tokenizer.decode(output[0][input_ids.shape[-1]:])
