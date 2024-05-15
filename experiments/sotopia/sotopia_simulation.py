import pandas as pd
import os
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

file_name = '/home/vqa/bell_hoon/Agents/PersonaAgent_final/experiments/sotopia/data/team_target_episodes_final.jsonl'

data = list()

with open(file_name, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data[0:5])
df = df[['scenario','agents_background','social_goals']]

model_directory = '/home/vqa/model-weights/llama3/infp_cleaned/checkpoint-400'
cache_dir = "/home/vqa/model-weights/llama3"

tokenizer = AutoTokenizer.from_pretrained(model_directory, cache_dir=cache_dir, use_cache=True)
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir, 
    use_cache=True,    
)

model1, tokenizer1 = model, tokenizer
model2, tokenizer2 = model, tokenizer


def generate_dialogue(model1, tokenizer1, model2, tokenizer2, prompt_a, prompt_b, mbti_a, mbti_b, max_length=1000, max_turns=10):
    dialogue = []
    next_speaker = 1
    messages_a = []
    messages_a.append({"role": "system", "content": f"You have a {mbti_a} personality"})
    messages_b = []
    messages_b.append({"role": "system", "content": f"You have a {mbti_b} personality"})
    for turn in range(max_turns):
        if next_speaker == 1:
            speaker_tag = "Person A: "
            model, tokenizer = model1, tokenizer1
        else:
            speaker_tag = "Person B: "
            model, tokenizer = model2, tokenizer2

        prompt_with_speaker_tag = speaker_tag
        input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
        ).to(model.device)

        terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = model.generate(input_ids,
        max_length=128, 
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        use_cache=True
        )
        response = output[0][input_ids.shape[-1]:]
        message = tokenizer.decode(response, skip_special_tokens=True)

        # 대화 추가
        clean_message = message.strip()
        if clean_message:
            dialogue.append(speaker_tag + clean_message)
            prompt = clean_message  # 가장 최근의 대화만을 prompt로 업데이트

        # 스피커 변경
        next_speaker = 1 if next_speaker == 2 else 2

    return dialogue


scenarios = []
background_A = []
background_B = []
goals_A = []
goals_B = []
A_secret = []
B_secret = []

for index, row in df.iterrows():
    scenario = row['scenario']
    social_goals = row['social_goals']
    agents_background = row['agents_background']

    goals = list(social_goals.values())
    backgrounds = list(agents_background.values())

    # <extra_info>와 <strategy_hint> 태그 제거
    goal_1 = re.sub(r'\s*<extra_info>.*?</extra_info>\s*', '', goals[0])
    goal_1 = re.sub(r'\s*<strategy_hint>.*?</strategy_hint>\s*', '', goal_1)

    goal_2 = re.sub(r'\s*<extra_info>.*?</extra_info>\s*', '', goals[1])
    goal_2 = re.sub(r'\s*<strategy_hint>.*?</strategy_hint>\s*', '', goal_2)

    # 괄호도 제거
    goal_1 = re.sub(r'\s*\(\)', '', goal_1)
    goal_2 = re.sub(r'\s*\(\)', '', goal_2)

    for agent, background in agents_background.items():
        background_info = re.split('Personality and values description:', background)[0].strip()
        secret_info = re.search("secrets: (.*)", background).group(1)

        if agent == list(agents_background.keys())[0]:
            background_A.append(background_info)
            A_secret.append(secret_info)
        else:
            background_B.append(background_info)
            B_secret.append(secret_info)

    scenarios.append(scenario)
    goals_A.append(goal_1)
    goals_B.append(goal_2)

new_df = pd.DataFrame({
    'scenario': scenarios,
    'A_goal': goals_A,
    'B_goal': goals_B,
    'A_background': background_A,
    'B_background': background_B,
    'A_secret': A_secret,
    'B_secret': B_secret
})


new_df['dialogue'] = ""

mbti_A = 'INFP'
mbti_B = 'INFP'
for index, row in new_df.iterrows():
    prompt_A = f'''
    You will be given a scenario.    
    Within the given scenario, you must try to achieve the given goal by interacting with another person. 
    While interacting, you have to a secret that you should keep from the other person knowing it. The sceneario, goal, and secret is presented as below:
    Scenario: {row['scenario']}
    Goal: {row['A_goal']}
    Secret: {row['A_secret']}
    Interact with the other person keeping in mind that you have {mbti_A} personality
    '''

    prompt_B = f'''
    You will be given a scenario.    
    Within the given scenario, you must try to achieve the given goal by interacting with another person. 
    While interacting, you have to a secret that you should keep from the other person knowing it. The sceneario, goal, and secret is presented as below:
    Scenario: {row['scenario']}
    Goal: {row['B_goal']}
    Secret: {row['B_secret']}

    You wi
    Interact with the other person keeping in mind that you have {mbti_B} personality
    '''
    
    dialogue = generate_dialogue(model1, tokenizer1, model2, tokenizer2, prompt_A, prompt_B, mbti_a, mbti_b)
    new_df.at[index, 'dialogue'] = " ".join(dialogue)

new_dict = new_df.to_dict()
with open('experiments/sotopia/outputs/sotopia_output.json', 'w') as f:
    json.dump(new_dict, f, indent=4)



