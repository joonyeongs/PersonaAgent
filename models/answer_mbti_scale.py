import os
import json
import time
import csv 
from model import *


mbti_questions = []

with open("data/mbti_data/16personality_questions_eng.csv", mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)    
    for row in reader:
        mbti_questions.append(row["Questions"]) 

mbti_questionnaire_ans = list()

models = ['llama2', 'llama3']
print(models)
idx = int(input('choose the model'))


chosen_model = models[idx]

for question in mbti_questions:
    system_prompt = '''You are an INFP.'''


    prompt = f'''You are tasked with responding to a statement from an MBTI (Myers-Briggs Type Indicator) questionnaire.

     Answer the question in the form of an integer from 1~7. If you agree with the statement, choose a number between 5~7. The more you agree with the statement, the larger the number you should choose.
     If you are neutral about it, choose 4. Else, if you disagree, choose a number between 1~3. The more you disagree, the smaller the number you will choose. 
     Your chosen number should reflect a nuanced understanding of your personality traits and preferences. The scale allows for a range of agreement or disagreement, enabling you to express subtle preferences and inclinations.
    Please consider each statement carefully and provide your response based on how much you agree or disagree with the statement. Remember, there are no right or wrong answers, only preferences that reflect different ways people perceive the world and make decisions. 

    No decimals, always an integer from 1 to 7.
    The question you have to answer is this:
    "{question}"

    '''
    
    if chosen_model == 'llama2':
        answer = llama2(system_prompt, prompt)
    elif chosen_model == 'llama3':
        answer = llama3(system_prompt, prompt)

    mbti_questionnaire_ans.append({"question": question, "answer": answer})
    #print(answer)
    print(f'finished')


with open(f"outputs/mbti/{chosen_model}_infp_mbti.json", "w") as f:
    json.dump(mbti_questionnaire_ans, f, indent=4)






