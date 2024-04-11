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

def create_pairs(dialogue):
    pairs = []
    for i in range(0, len(dialogue), 2):
        if i+1 < len(dialogue):
            pairs.append({"prompt": dialogue[i], "chosen": dialogue[i+1], })

    return pairs


def preprocess_data(dataset):
    prompt_and_chosen_data = []

    for data in dataset:
        if data["dialogue"][0].find("\"User") == -1:
            data["dialogue"] = data["dialogue"][1:]
        
        cleaned_dialogue = [clean_data(string_to_clean) for string_to_clean in data["dialogue"]]  
      
        prompt_and_chosen = create_pairs(cleaned_dialogue)
        prompt_and_chosen_data.extend(prompt_and_chosen)
    
    return prompt_and_chosen_data
