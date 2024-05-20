import json
import copy


with open('/home/vqa/bell_hoon/Agents/PersonaAgent_final/data/machine_mindset/dpo_execution_p_j.json', 'r') as f:
   dataset = json.load(f)

dataset = dataset[0:5]
def switch_chosen_rejected(dataset):
   for data in dataset:
      temp = copy.deepcopy(data['rejected'])
      data['rejected'] = data['chosen']
      data['chosen'] = temp
   return dataset

print((dataset[3:4]))
print(switch_chosen_rejected(dataset[3:4]))

