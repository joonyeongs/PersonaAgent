import json
import csv

# Load JSON data
output_dir = "/home/vqa/data/outputs/beavertails"

integrated_outputs = {"none": [], "ENTJ": [], "ESTJ": [], "INFJ": [], "INFP": []}
for mbti in ['none', 'ENTJ', 'ESTJ', 'INFJ', 'INFP']:
    eval_output_dir = f"{output_dir}/{mbti}_eval.json"

    with open(eval_output_dir, "r") as f:
        outputs = json.load(f)

    cnt = 0
    for output in outputs:
        #is_correct = 1 if output['label'] == output['chosen_answer'] else 0
        is_correct = 1 if output['gpt_evaluated'].startswith('flagged') else 0
        cnt += is_correct
        integrated_outputs[mbti].append(is_correct)
    print(f"Accuracy for {mbti} is {cnt}")

    



with open('/home/vqa/data/outputs/beavertails/output.csv', 'w', newline='') as csvfile:
    fieldnames = ['none', 'ENTJ', 'ESTJ', 'INFP', 'INFJ']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    dataset_length = len(integrated_outputs['none'])
    for i in range(dataset_length):
        # Create a dictionary to store results for this sample
        row = {model: integrated_outputs[model][i] for model in fieldnames}
        # Write the row to the CSV file
        writer.writerow(row)

print("CSV file has been created.")
