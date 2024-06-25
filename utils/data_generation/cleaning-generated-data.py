import json
import re

# JSON 데이터를 불러오기 위한 함수
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# JSON 데이터를 내보내기 위한 함수
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def cleansing_text(text):
    # 문장 분리
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    cleaned_sentences = []
    for sentence in sentences:
        # That~시작 문장 삭제
        if sentence.startswith('that'):
            continue
        # 단어 삭제 
        sentence = re.sub(r'^(Hey,|Hmm,|Well,|Wow,|Ah,|Absolutely!)', '', sentence).strip()
        cleaned_sentences.append(sentence)
    
    # 다시 문장 합치기 
    cleaned_text = ' '.join(cleaned_sentences)
    return cleaned_text

# 데이터 클렌징 함수
def cleansing_function(data):
    for item in data:
        if 'prompt' in item and item['prompt']:
            item['prompt'] = cleansing_text(item['prompt'])


input_file_path = '/home/joonyeongs/persona-agent/data/generated_data/entj_pair_data_cleaned.json'
output_file_path = '/home/joonyeongs/persona-agent/data/generated_data/entj_pair_data_cleaned.json'

data = load_json(input_file_path)
cleansing_function(data)
save_json(data, output_file_path)
