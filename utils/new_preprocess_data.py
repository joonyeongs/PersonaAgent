import json

def clean_data(string_to_clean):
    # Remove \"INFP\": and \"User\":
    start_index = string_to_clean.find(":")
    string_to_clean = string_to_clean[start_index + 1:]

    # Remove *chuckles*
    first_star_index = string_to_clean.find('*')
    second_star_index = string_to_clean.find('*', first_star_index + 1)
    if second_star_index != -1 and first_star_index == 0:
        string_to_clean = string_to_clean[second_star_index+1:]
    elif second_star_index != -1 and first_star_index > 0:
        string_to_clean = string_to_clean[0:first_star_index] + string_to_clean[second_star_index+1:]
    else:
        pass
    # Remove \" at the end of the string
    string_to_clean = string_to_clean.strip()
    string_to_clean = string_to_clean.lstrip('\"')
    string_to_clean = string_to_clean.rstrip('\"')

    return string_to_clean

with open('data/generated_data/infj/infj_character_no_situation_pair_data.json', 'r') as f:
    character_no_situation = json.load(f)

with open('data/generated_data/infj/infj_character_situation_pair_data.json', 'r') as f:
    character_situation = json.load(f)

with open('data/generated_data/infj/infj_feature_situation_pair_data.json', 'r') as f:
    feature_situation = json.load(f)

with open('data/generated_data/infj/infj_feature_no_situation_pair_data.json', 'r') as f:
    feature_no_situation = json.load(f)

total_data = character_no_situation + character_situation + feature_situation + feature_no_situation

cleaned_data = []
print('total_len: ', len(total_data))
count = 0

for data in total_data:
    cleaned_dict = {}#{key: value for key, value in data.items() if key not in ["chosen", "rejected"]}
    keys = ['chosen', 'rejected']    
    for key in keys:
        data[key] =  [item for item in data[key] if len(item) >= 5]      ## "{" 이런거 있어서 이런거 버리는애들
    
    if len(data['chosen']) < len(data['rejected']):
        count += 1
    
    if data['chosen'][0].startswith("Here is the dialogue") or "{" in data['chosen'][0]:
        data['chosen'].pop(0)
    if data['rejected'][0].startswith("Here is the dialogue") or "{" in data['rejected'][0]:
            data['rejected'].pop(0)

    #data["chosen"] = [{"content": clean_data(string_to_clean), "role": "user" if i % 2 == 0 else "INFP"} for i, string_to_clean in enumerate(data["chosen"])]    
    #data["rejected"] = [{"content": clean_data(string_to_clean), "role": "user" if i % 2 == 0 else "INFP"} for i, string_to_clean in enumerate(data["rejected"])]
    data["chosen"] = [clean_data(string_to_clean) for string_to_clean in data["chosen"]]  
    data["rejected"] = [clean_data(string_to_clean) for string_to_clean in data["rejected"]]    
    
    for i in range(0, min(len(data['chosen']), len(data['rejected'])), 2):
        if i+1 < min(len(data['chosen']), len(data['rejected'])):           ## out of bounds 방지
            if data["chosen"][0+i] == data["rejected"][0+i]:                ## rejected와 chosen의 user 발화가 일치하다면~
                cleaned_dict["prompt"] = data["chosen"][0+i]
            cleaned_dict["chosen"] = data["chosen"][1+i]
            cleaned_dict["rejected"] = data["rejected"][1+i]    
            if 'prompt' in cleaned_dict.keys():
                cleaned_data.append(cleaned_dict)                           ## prompt가 생성될 경우에만 data append
    

print('count', count)
print('length of data', len(cleaned_data))
with open('data/generated_data/infj/infj_total_data.json', 'w') as f:
    json.dump(cleaned_data, f, indent=4)
