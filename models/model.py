from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import torch


HUGGINGFACE_TOKEN = 'your_hugginface_token'

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_directory):
    device = get_device_map()
    model, tokenizer = FastLanguageModel.from_pretrained(
          model_name = model_directory,
          device_map=device,
          max_seq_length = 2048, ## doesn't matter
          dtype = torch.bfloat16, 
          load_in_4bit = False, 
          token = HUGGINGFACE_TOKEN,
          cache_dir="/home/vqa/model-weights/llama3",
          use_cache=True,
      )
    FastLanguageModel.for_inference(model)
    print(model.name_or_path)
    return tokenizer, model


def model_generate(system, desired_prompt, tokenizer, model, multi_turn = False, messages=[]):

    if not multi_turn:
        messages = []
    EOS_TOKEN = tokenizer.eos_token
    
    if len(messages) == 0 or messages[0]["role"] != "system":
         print('system message appended')
         messages.append({"role": "system", "content": system})   
    messages.append({"role": "user", "content": desired_prompt})
    #messages.append({"role": "assistant", "content": 'My score to the given question is : '})   

    '''
    print(tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=False,
    ))
    '''
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
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



'''
system = 'Answer the dialogue  '

tokenizer, model = load_model('/home/vqa/model-weights/llama3/annotated/infp/more/checkpoint-40')
prompt = input()
while prompt != '\n':
    print(model_generate(system, prompt, tokenizer, model))
    prompt = input('ask INFP: ')

'''
'''You are an INFP, which is known for known for their deep empathy, creativity, introspection, open-mindness, and preference for meaningful connections over superficial interactions, while being an introvert. 
  INFPs are often seen as idealistic and passionate, with a unique ability to see the good in people and situations. INFPs are adaptable and flexible, often exploring various possibilities rather than committing to a rigid plan.
  You will be given a part of a dialogue. Respond the given dialogue keeping in mind that you are an INFP. '''