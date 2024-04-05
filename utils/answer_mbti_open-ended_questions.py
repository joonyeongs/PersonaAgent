from dotenv import load_dotenv

import os
import anthropic
import json
import time
import csv 

with open("data/generated_data/mbti_openended_questions.json", "r") as f:
    questions = json.load(f)

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

answers_to_openended_questions = list()


for question in questions:
    system_prompt='''
    Create in the following format, where your answer to the given question is surrounded by quotes:
    "Answer": "your answer"

    Following this format is very very important.
    '''

    prompt = f'''You are tasked with answering an open-ended question that has been crafted to explore various aspects of personality, behaviors, and preferences. This question is designed to prompt reflection and self-analysis, inviting you to share insights into your own experiences, feelings, and thoughts.

    Your response should aim to capture the complexity of your preferences in a manner that is reflective, thoughtful, and detailed. Please consider each statement carefully and provide your response based on how you think about the given question.
    Ensure to recognize the complexity of human behvior and personality by offering nuanced answers, as well as being reflective of your personality, honest, and descriptive.

    The question you have to answer is this:
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
    answers_to_openended_questions.append({"question": question, "answer": answer})
    print(f'finished')


with open("data/generated_data/mbti_open-ended_ans.json", "w") as f:
    json.dump(answers_to_openended_questions , f, indent=4)






