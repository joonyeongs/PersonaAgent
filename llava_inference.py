from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the path to your local model directory
model_directory = '/workspace/model-weights/checkpoint-48'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

model.to('cuda:5')

# Example usage

desired_prompt = input()

prompt = f'''<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{desired_prompt} [/INST]
'''


input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:5')

# Generate a response
output = model.generate(input_ids, max_length=500)
print(tokenizer.decode(output[0], skip_special_tokens=True))
