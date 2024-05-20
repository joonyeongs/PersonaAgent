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


#mbti_questions = mbti_questions[-13:]
mbti_questionnaire_ans = list()


mbtis = ['infp', 'infj', 'estj', 'entj', 'infp', 'infj', 'estj', 'entj']
save_path = ['machine_infp', 'infj', 'estj', 'entj', 'low_infp', 'low_infj', 'low_estj', 'low_entj']
model_paths = [        
        '/home/vqa/model-weights/llama3/machine_mindset/infp/checkpoint-54',
        '/home/vqa/model-weights/llama3/infj_cleaned/checkpoint-111',
        '/home/vqa/model-weights/llama3/estj_cleaned/checkpoint-117',
        '/home/vqa/model-weights/llama3/entj_cleaned/checkpoint-249',
        '/home/vqa/model-weights/llama3/low_lr/infp_cleaned/checkpoint-240',
        '/home/vqa/model-weights/llama3/low_lr/infj_cleaned/checkpoint-111',
        '/home/vqa/model-weights/llama3/low_lr/estj_cleaned/checkpoint-117',
        '/home/vqa/model-weights/llama3/low_lr/entj_cleaned/checkpoint-249',    
    ]
#idx = int(input('your desired mbti idx here: '))

chosen_model = 'llama3'
#chosen_path = model_paths[idx]


for model_path , mbti, path in zip(model_paths[0:1], mbtis[0:1], save_path[0:1]):
    print('loading ', model_path)
    tokenizer, model = load_model(model_path)
    for question in mbti_questions:
        system_prompt = f'''You are an {mbti.upper()}.'''

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
            answer = model_generate(system_prompt, prompt, tokenizer, model,)

        mbti_questionnaire_ans.append({"question": question, "answer": answer})
        #print(answer)
        print(f'finished')


    with open(f"outputs/mbti/official_metric/{chosen_model}_{path}_mbti.json", "w") as f:
        json.dump(mbti_questionnaire_ans, f, indent=4)






