from dotenv import load_dotenv
from joblib import Parallel, delayed

import os
import anthropic
import json
import time
import random
import sys

def generate_dialgue(character, situation):
    system_prompt='''
    Create in the following format where the conversation utterances are surrounded by quotes:
    "User": "Detailed utterance 1"
    "INFP": "Detailed utterance 2"

    In this conversation, "User" is an arbitrary person and "INFP" is the INFP character. Ensure that the subject of the dialogue follows the given situation. Assume that “User” and “INFP” have already been introduced and are familiar with each other. The conversation should be a back-and-forth dialogue. Each person should speak between 4 to 10 times. 
    Following this format is mandatory and very important. Do not mention the character's name.
    '''

    prompt = f'''Begin by exploring the broad characteristics of the INFP personality type, known for their deep empathy, creativity, introspection, and preference for meaningful connections over superficial interactions. INFPs are often seen as idealistic and passionate, with a unique ability to see the good in people and situations. They are driven by their values and seek to understand themselves and others on a profound level. INFPs are also known for their adaptability and open-mindedness, allowing them to navigate complex emotional landscapes with grace. 
    Given these traits, generate a dialogue between an arbitrary person and a character from a fictional story whose personality type is INFP and has following traits: 
    “{character}"

    The dialogue should be insightful and reflective of the character’s personality, showcasing their empathetic, introspective, and idealistic nature of INFP in different contexts. Ensure that the dialogues illustrate the character's nuanced approach to its trait. Aim for a natural progression in the dialogues, from introduction to a deeper exploration of the topic, effectively capturing the essence of the INFP personality through this specific lens. Don't mention the word "INFP" in the conversation.
    This conversation is happening in this situation:
    "{situation}"
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

    dialogue = message.content[0].text.splitlines()
    dialogue = list(filter(lambda x: x != "", dialogue))

    output = {"character": character, "scene": situation, "dialogue": dialogue}
    
    return output

# load character data
with open("data/seed_data/infp_character_profile_general.json", "r") as f:
    infp_data = json.load(f)

with open("data/seed_data/random_situations.json", "r") as f:
    situation_data = json.load(f)

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)


infp_dialogue_dataset = list()
for round in range(8):
    for i in range(0, len(infp_data), 3):
        print(f"Generating dialogue for character {i+1} to {i+3} of {len(infp_data)} in Round {round+1}")
        success = False
        error_count = 0
        situations = list(situation_data.values())

        while not success:
            try:
                outputs = Parallel(n_jobs=3, verbose=100, prefer="threads")(
                    delayed(generate_dialgue)(
                        character=character["profile"],
                        situation=random.choice(situations)
                    )
                    for character in infp_data[i: i+3]
                )
                for output in outputs:
                    infp_dialogue_dataset.append(output)
                success = True
            except Exception as e:
                print(e)
                error_count += 1
                if error_count >= 5:
                    with open("data/generated_data/infp_character_situation_dialogues.json", "w") as f:
                        json.dump(infp_dialogue_dataset, f, indent=4)
                    sys.exit("Too many errors, exiting")
                time.sleep(20)

with open("data/generated_data/infp_character_situation_dialogues.json", "w") as f:
    json.dump(infp_dialogue_dataset, f, indent=4)






