from dotenv import load_dotenv

import os
import anthropic
import json
import time

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

infp_situation_dataset = list()
for i in range(1):

    prompt = '''
    Personality:
    "INFPs are quiet and reserved. They prefer spending time alone or with a small group of close friends."

    Imagine 20 dialouge situations that describe INFP's personality only based on the information above.
    The situations should be described concisely, focusing on the background and without telling the details.
    Return in JSON format. "key" should be f"scene_{number}" and "value" should be the situation description.

    '''

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=1,
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    infp_situation_dataset.append(message.content[0].text.split('\n\n'))
    print(f'finished_{i}')


json_string = infp_situation_dataset[0]
json_text = json_string[0]
json_object = json.loads(json_text)

with open("infp_situation_data_sample.json", "w", encoding="utf-8") as json_file:
    json.dump(json_object, json_file, ensure_ascii=False, indent=4)