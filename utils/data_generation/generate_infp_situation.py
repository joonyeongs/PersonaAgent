from dotenv import load_dotenv

import os
import anthropic
import json
import time

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

with open("data/seed_data/mbti_profiles_infp.json", "r") as f:
    infp_data = json.load(f)

infp_situation_dataset = list()
for i in range(1):

    prompt = f'''
    Personality:
    "{infp_data[i]['text']}"

    Imagine 8 dialouge situations that describe INFP's personality only based on the information above.
    The situations should be described concisely, focusing on the background and without telling the details.
    Return in JSON format. "key" should be scene_[number] and "value" should be the situation description.

    '''

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=1,
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    print(message.content[0].text)
    scene_data = eval(message.content[0].text)
    infp_situation_dataset.append({**infp_data[i], "scene": scene_data})
    print(f'finished_{i}')


with open("infp_situation_data_sample.json", "w", encoding="utf-8") as json_file:
    json.dump(infp_situation_dataset, json_file, ensure_ascii=False, indent=4)