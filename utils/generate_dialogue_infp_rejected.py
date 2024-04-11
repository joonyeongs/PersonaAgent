from dotenv import load_dotenv
from joblib import Parallel, delayed

import os
import anthropic
import json
import time
import sys

def reverse_response(prompt, chosen):
    system_prompt='''
    Given a dialogue between an INFP individual and an arbitrary user, please invert the INFP's responses to reflect opposite personality traits.
    '''

    content = f'''Given a dialogue between an INFP individual and an arbitrary user, please invert the INFP's responses to reflect opposite personality traits.
=    The context of the dialogue remains the same, but the tone, approach, and underlying values in the responses from the INFP character should reflect these changes, ensuring a clear contrast to the original dialogue’s INFP characteristics.
    The user says: "{prompt}"
    The INFP character says: "{chosen}"

    Now, reverse the INFP character's response to reflect opposite personality traits. Answer in the format below:
    Reversed response: "The INFP character's reversed response."
    '''

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        system=system_prompt,
        max_tokens=1000,
        temperature=1,
        messages=[
            {"role": "user", "content": content}
        ]
    )

    rejected = message.content[0].text

    start_index = rejected.find("\"")
    end_index = rejected.rfind("\"")
    if start_index != -1 and end_index != -1:
        rejected = rejected[start_index + 1:end_index]
    
    return rejected



with open("data/generated_data/infp_character_situation_prompt_chosen.json", "r") as f:
    dataset = json.load(f)

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

start = time.time()

'''
for i in range(0, len(prompt_chosen), 5):
    print(f"Generating dialogue for character {i+1} to {i+5} of {len(prompt_chosen)}")
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
'''
pair_data = []
for i, data in enumerate(dataset):
    print(f"Generating dialogue for character {i+1} of {len(dataset)}")
    prompt, chosen = data["prompt"], data["chosen"]

    success = False
    error_count = 0
    while not success:
        try:
            rejected = reverse_response(prompt, chosen)
            success = True
        except Exception as e:
            error_count += 1
            if error_count >= 5:
                with open("data/generated_data/infp_pair_data.json", "w") as f:
                    json.dump(pair_data, f, indent=4)
                sys.exit("Too many errors, exiting")
            time.sleep(20)

    pair_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if len(pair_data) > 9:
        break


with open("data/generated_data/infp_pair_data.json", "w") as f:
    json.dump(pair_data, f, indent=4)

end = time.time()
print(f"Time taken: {end-start}")