import json
import re

with open("outputs/mbti/llama3_infp_mbti.json", "r", encoding='utf-8') as f:
  dataset = json.load(f)


count = 0
pattern = r'[1-7]'
for info in dataset:
  match = re.search(pattern, info['answer'])
  if match:
    print(info['question'], match.group(0))
    count += 1

print(len(dataset), count)







'''
for i in range(len(dataset)):
  for key in dataset[i].keys():
    if 'As an INFP' in dataset[i][key]:
        print(dataset[i][key])
    dataset[i][key] = dataset[i][key].replace('As an INFP,', '')
    dataset[i][key] = dataset[i][key].replace('As an INFP', '')
    dataset[i][key] = dataset[i][key].replace('as an INFP', '')
    dataset[i][key] = dataset[i][key].replace('As an introvert,', '')
    dataset[i][key] = dataset[i][key].replace('as an introvert', '')
    if 'As an INFP' in dataset[i][key]:
        print(dataset[i][key])

with open("data/generated_data/infp_pair_data_cleaned.json", "w") as f:
  json.dump(dataset, f, indent=4)
'''