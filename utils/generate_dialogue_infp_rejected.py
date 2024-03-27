from dotenv import load_dotenv

import os
import anthropic
import json
import time

def concatenate_dialogues(dialogues):
    return '\n'.join(dialogues)

with open("data/generated_data/infp_feature_dialogues.json", "r") as f:
    infp_dialogues = json.load(f)

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

infp_dialogue_dataset = list()
for i in range(1):
    dialogue_to_invert = concatenate_dialogues(infp_dialogues[i]["dialogue"])

    system_prompt='''
    Create in the following format where the conversation utterances are surrounded by quotes:
    "User": "Detailed utterance 1"
    "INFP": "Detailed utterance 2"

    In this conversation, "User" is an arbitrary person and "INFP" is the INFP character. They have already been introduced and are familiar with each other. The conversation should be a back-and-forth dialogue. Each person should speak between 4 to 10 times.
    Following this format is mandatory and very important.
    '''

    prompt = f'''Given a dialogue between an INFP individual and an arbitrary user, please invert the INFP's responses to reflect opposite personality traits.
    Adjust the arbitrary user’s responses as necessary to maintain a coherent conversation that highlights the shift in the INFP character’s personality traits. 
    The context of the dialogue remains the same, but the tone, approach, and underlying values in the responses from the INFP character should reflect these changes, ensuring a clear contrast to the original dialogue’s INFP characteristics.
    The dialogue that you should invert is:
    "{dialogue_to_invert}"
    '''
    success = False

    while success == False:
        try:
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                system=system_prompt,
                max_tokens=1000,
                temperature=1,
                messages=[
                    {"role": "user", "content": f"{prompt}"}
                ]
            )
            success = True
        except Exception as e:
            print(e)
            time.sleep(20)
    inverted_dialogue = message.content[0].text.split("\n")
    infp_dialogue_dataset.append({"feature":infp_dialogues[i]["feature"], "chosen": infp_dialogues[i]["dialogue"], "rejected": inverted_dialogue})
    print(f'finished_{i}')


with open("data/generated_data/infp_pair_data.json", "w") as f:
    json.dump(infp_dialogue_dataset, f, indent=4)