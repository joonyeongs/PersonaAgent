from dotenv import load_dotenv

import os
import anthropic
import json
import time
import csv 

mbti_questions = []

with open("data/seed_data/16personality_questions_eng.csv", mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        mbti_questions.append(row["Questions"]) 

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

mbti_questionnaire_ans = list()

for question in mbti_questions:
    system_prompt='''
    Create in the following format:
    "Answer": "your answer, a number between 1 and 7"

    You are to provide a response on a scale of 1 to 7, where 1 signifies "Strongly disagree" and 7 signifies "Strongly agree." Intermediate values should reflect varying degrees of agreement or disagreement, with a response of 4 representing a neutral position or ambivalence.
    There is no need for any additional explanation. Following this format is very very important.
    '''

    prompt = f'''You are tasked with responding to a statement from an MBTI (Myers-Briggs Type Indicator) questionnaire. 

    Your responses should reflect a nuanced understanding of your personality traits and preferences. The scale allows for a range of agreement or disagreement, enabling you to express subtle preferences and inclinations.
    Please consider each statement carefully and provide your response based on how much you agree or disagree with the statement. Remember, there are no right or wrong answers, only preferences that reflect different ways people perceive the world and make decisions. Your responses should aim to capture the complexity of these preferences in a manner that is reflective, thoughtful, and as detailed as the scale allows.

    The statement you have to answer is this:
    "{question}"
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
    answer = message.content[0].text.split("\n\n")
    mbti_questionnaire_ans.append({"question": question, "answer": answer})
    print(f'finished')


with open("data/generated_data/infp_feature_dialogues.json", "w") as f:
    json.dump(mbti_questionnaire_ans, f, indent=4)






