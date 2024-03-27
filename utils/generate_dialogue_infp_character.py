from dotenv import load_dotenv

import os
import anthropic
import json
import time

with open("data/seed_data/mbti_profiles_infp.json", "r") as f:
    infp_data = json.load(f)

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

infp_dialogue_dataset = list()
for i in range(3):
    my_character = infp_data[i]
    prompt=f'''Begin by exploring the broad characteristics of the INFP personality type, known for their deep empathy, creativity, introspection, and preference for meaningful connections over superficial interactions. INFPs are often seen as idealistic and passionate, with a unique ability to see the good in people and situations. They are driven by their values and seek to understand themselves and others on a profound level. INFPs are also known for their adaptability and open-mindedness, allowing them to navigate complex emotional landscapes with grace.

    Given these traits, generate a dialogue between an arbitrary person (“user”) and a fictional character (“assistant”) whose personality type is INFP and has following traits:
    “{my_character['text']}"
    The dialogue should be insightful and reflective of the character’s personality, showcasing their empathetic, introspective, and idealistic nature of INFP in different contexts. Ensure that the dialogues illustrate the character's nuanced approach to its trait. Aim for a natural progression in the dialogues, from introduction to a deeper exploration of the topic, effectively capturing the essence of the INFP personality through this specific lens. Ensure that the unique worldbuilding and concept of the fiction is not reflected in the dialogue. The dialogue has to be about general, realistic situation.
    '''
    system_prompt='''
    Create in the following format:
    {"role": "user", "content": ""},
    {"role": "assistant", "content": ""}

    In this conversation, "user" is an arbitrary person and "assistant" is the character. The dialogue should be in the "content". Each person should speak between 4 to 10 times.

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
    dialogue = message.content[0].text.split("\n\n")
    infp_dialogue_dataset.append({"character": my_character['text'], "dialogue": dialogue})
    print(f'finished_{i}')



with open("data/generated_data/infp_character_dialogues.json", "w") as f:
    json.dump(infp_dialogue_dataset, f, indent=4)