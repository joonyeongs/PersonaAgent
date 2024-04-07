import json

def clean_data(string_to_clean):
    # Remove \"INFP\": and \"User\":
    start_index = string_to_clean.find(":")
    string_to_clean = string_to_clean[start_index + 1:]

    # Remove *chuckles*
    first_star_index = string_to_clean.find('*')
    second_star_index = string_to_clean.find('*', first_star_index + 1)
    if second_star_index != -1:
        string_to_clean = string_to_clean[second_star_index+1:]

    # Remove \" at the end of the string
    string_to_clean = string_to_clean.strip()
    string_to_clean = string_to_clean.lstrip('\"')
    string_to_clean = string_to_clean.rstrip('\"')

    return string_to_clean

with open('data/generated_data/infp_pair_data_character_no_situation.json', 'r') as f:
    character_no_situation = json.load(f)

with open('data/generated_data/infp_pair_data_character_situation.json', 'r') as f:
    character_situation = json.load(f)

with open('data/generated_data/infp_feature_situation_pair_data.json', 'r') as f:
    feature_situation = json.load(f)

total_data = character_no_situation + character_situation + feature_situation

cleaned_data = []

for data in total_data:
    cleaned_dict = {}#{key: value for key, value in data.items() if key not in ["chosen", "rejected"]}

    
    #data["chosen"] = [{"content": clean_data(string_to_clean), "role": "user" if i % 2 == 0 else "INFP"} for i, string_to_clean in enumerate(data["chosen"])]    
    #data["rejected"] = [{"content": clean_data(string_to_clean), "role": "user" if i % 2 == 0 else "INFP"} for i, string_to_clean in enumerate(data["rejected"])]
    data["chosen"] = [clean_data(string_to_clean) for string_to_clean in data["chosen"]]  
    data["rejected"] = [clean_data(string_to_clean) for string_to_clean in data["rejected"]]    

    cleaned_dict["prompt"] = data["chosen"][0]
    cleaned_dict["chosen"] = data["chosen"][1]
    cleaned_dict["rejected"] = data["rejected"][1]

    
    if cleaned_dict["prompt"]:
        cleaned_data.append(cleaned_dict)
 
with open('data/generated_data/infp_total_data.json', 'w') as f:
    json.dump(cleaned_data, f, indent=4)
