from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the path to your local model directory
model_directory = '/home/vqa/00_backup/model-weights/llama2/infp/checkpoint-400'


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_directory, cache_dir="/home/vqa/00_backup/model-weights/llama2", use_cache=True)
model = AutoModelForCausalLM.from_pretrained(model_directory, cache_dir="/home/vqa/00_backup/model-weights/llama2", use_cache=True)
system_prompt = 'You have a INFP personality'
# Example usage
while True:
    desired_prompt = input('write your desired input here: ')
    if desired_prompt == '\n':
        break
        
    prompt = f'<s>[INST] <<SYS>>\n You are an INFP\n<</SYS>>\n{desired_prompt} [/INST]' #prompt format for llama2
    #prompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{desired_prompt}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>' 
    


    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

    # Generate a response
    output = model.generate(input_ids, max_length=500)
    print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))
