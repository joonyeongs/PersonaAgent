from dotenv import load_dotenv

import os
import anthropic
import json
import time
import random

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
for i in range(len(infp_data)):
    character = infp_data[i]["profile"]

    # TODO: Actually use the situation data
    situations = list(situation_data.values())

    system_prompt='''
    Create in the following format where the conversation utterances are surrounded by quotes:
    "User": "Detailed utterance 1"
    "INFP": "Detailed utterance 2"

    In this conversation, "User" is an arbitrary person and "INFP" is the INFP character. Ensure that the subject of the dialogue follows the given situation. Assume that “User” and “INFP” have already been introduced and are familiar with each other. The conversation should be a back-and-forth dialogue. Each person should speak between 4 to 10 times. 
    Following this format is mandatory and very important. Do not mention the character's name.
    '''

    for _ in range(8):
        situation = random.choice(situations)
        prompt = f'''Begin by exploring the broad characteristics of the INFP personality type, known for their deep empathy, creativity, introspection, and preference for meaningful connections over superficial interactions. INFPs are often seen as idealistic and passionate, with a unique ability to see the good in people and situations. They are driven by their values and seek to understand themselves and others on a profound level. INFPs are also known for their adaptability and open-mindedness, allowing them to navigate complex emotional landscapes with grace. 

        Given these traits, generate a dialogue between an arbitrary person and a character from a fictional story whose personality type is INFP and has following traits: 
        “{character}"
        
        The dialogue should be insightful and reflective of the character’s personality, showcasing their empathetic, introspective, and idealistic nature of INFP in different contexts. Ensure that the dialogues illustrate the character's nuanced approach to its trait. Aim for a natural progression in the dialogues, from introduction to a deeper exploration of the topic, effectively capturing the essence of the INFP personality through this specific lens. Don't mention the word "INFP" in the conversation.

        This conversation is happening in this situation:
        "{situation}"
        '''
        success = False

        error_count = 0
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
                error_count += 1
                if error_count >= 5:
                    break
                time.sleep(20)
        dialogue = message.content[0].text.split("\n\n")
        infp_dialogue_dataset.append({"character": character, "scene": situation, "dialogue": dialogue})
        print(f'finished_{i}')


with open("data/generated_data/infp_character_situation_dialogues.json", "w") as f:
    json.dump(infp_dialogue_dataset, f, indent=4)






