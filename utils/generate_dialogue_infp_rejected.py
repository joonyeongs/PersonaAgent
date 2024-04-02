from dotenv import load_dotenv
from joblib import Parallel, delayed

import os
import anthropic
import json
import time
import sys

def concatenate_dialogues(dialogues):
    return '\n'.join(dialogues)

def generate_dialgue(infp_dialogue):
    dialogue_to_invert=concatenate_dialogues(infp_dialogue["dialogue"]),
    start_of_dialogue=infp_dialogue["dialogue"][0]
    character = infp_dialogue["character"]
    situation = infp_dialogue["scene"]

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

    Start the conversation with the following response from the User:
    "{start_of_dialogue}"
    '''

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        system=system_prompt,
        max_tokens=1000,
        temperature=1,
        messages=[
        {"role": "user", "content": f"{prompt}"}
        ]
    )

    inverted_dialogue = message.content[0].text.splitlines()
    inverted_dialogue = list(filter(lambda x: x != "", inverted_dialogue))

    output = {"character": character, "scene": situation, "chosen": infp_dialogue["dialogue"], "rejected": inverted_dialogue}
    
    return output



with open("data/generated_data/infp_character_situation_dialogues.json", "r") as f:
    infp_dialogues = json.load(f)

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

infp_dialogue_dataset = list()

start = time.time()

for i in range(0, len(infp_dialogues), 5):
    print(f"Generating dialogue for character {i+1} to {i+5} of {len(infp_dialogues)}")
    success = False
    error_count = 0

    while not success:
        try:
            outputs = Parallel(n_jobs=3, verbose=100, prefer="threads")(
                delayed(generate_dialgue)(
                    infp_dialogue=infp_dialogue
                )
                for infp_dialogue in infp_dialogues[i: i+5]
            )
            for output in outputs:
                infp_dialogue_dataset.append(output)
            success = True
        except Exception as e:
            print(e)
            error_count += 1
            if error_count >= 5:
                with open("data/generated_data/infp_pair_data.json", "w") as f:
                    json.dump(infp_dialogue_dataset, f, indent=4)
                sys.exit("Too many errors, exiting")
            time.sleep(20)



with open("data/generated_data/infp_pair_data.json", "w") as f:
    json.dump(infp_dialogue_dataset, f, indent=4)

end = time.time()
print(f"Time taken: {end-start}")