from dotenv import load_dotenv

import os
import anthropic
import json
import time
import csv 

mbti_statements = []

with open("data/seed_data/16personality_questions_eng.csv", mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        mbti_statements.append(row["Questions"]) 

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

mbti_openended_questions = list()

for statement in mbti_statements:
    system_prompt='''
    Create in the following format, where the newly created open-ended question is surrounded by quotes:
    "newly created open-ended question"

    There is no need for any additional explanation. Following this format is very very important.
    '''

    prompt = f'''Your task is to transform a given statement into an open-ended question. The statement will relate to personal behaviors, preferences, or traits, similar to the kind used in personality assessments or introspective exercises. Your goal is to rephrase each statement as a question in a way that invites the responder to reflect upon and share their experiences, feelings, or thoughts related to the statement.

    For instance, if presented with the statement, "You regularly make new friends," your response should be, "Do you regularly make new friends?" 

    This conversion process involves adding an interrogative component to the beginning of the statement, effectively turning it into a question that prompts personal reflection.

    The statement you have to convert is this:
    "{statement}"
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
    openended_qustion = message.content[0].text.split("\n\n")
    mbti_openended_questions.append({"original statement": statement, "open-ended question": openended_qustion})
    print(f'finished')


with open("data/generated_data/infp_feature_dialogues.json", "w") as f:
    json.dump(mbti_openended_questions, f, indent=4)






