import json
import os
import re

paths = os.listdir('data/machine_mindset')

def has_chinese(text):
    if '\u4e00' <= text[:6] <= '\u9fff':
        return True
    return False

for path in paths:
    with open('data/machine_mindset/'+path, 'r', encoding='utf-8', ) as f:
        dataset = json.load(f)
    for data in dataset:
        if not has_chinese(data['instruction']) and not has_chinese(data['input']):
            data['prompt'] = data['instruction']
            data['chosen'] = data['output'][0]
            data['rejected'] = data['output'][1]
            del data['instruction'], data['output'], data['input']
        else:
            del data        
        

    with open('data/machine_mindset/'+path, 'w') as f:
        json.dump(dataset, f, indent=4)

