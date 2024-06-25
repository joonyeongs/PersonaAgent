import os
import pandas as pd
import time
from tqdm import tqdm


def convert_results(result, column_header):
    result = result.strip()  # Remove leading and trailing whitespace
    try:
        result_list = [int(element.strip()[-1]) for element in result.split('\n') if element.strip()]
    except:
        result_list = ["" for element in result.split('\n')]
        print(f"Unable to capture the responses on {column_header}.")
        
    return result_list


def example_generator(questionnaire, args):
    testing_file = args.testing_file
    model = args.model


    # Read the existing CSV file into a pandas DataFrame
    testing_df = pd.read_csv(testing_file)
    headers = testing_df.columns.tolist()

    questions_list = {f'order-{header[-1]}': '\n'.join(testing_df[header].astype(str)) for header in headers if header.startswith("question")}
    testing_list = [{"key":header, "scenario":testing_df[header].iloc[0]} for header in headers if not header.startswith(("question", "order"))]
    
    for test_case in tqdm(testing_list):
        result = ''
        order_key = test_case["key"].split('_')[-1]
        prompt = questions_list[order_key].replace('SCENARIO', test_case["scenario"])
        
        ### Generate the response
        if model == 'text-davinci-003':
            inputs = prompt
            result = completion(model, inputs)
        elif model in ['gpt-3.5-turbo', 'gpt-4']:
            inputs = [
                {"role": "system", "content": questionnaire["inner_setting"]},
                {"role": "user", "content": prompt}
            ]
            result = chat(model, inputs)
        else:
            raise ValueError("The model is not supported or does not exist.")
        ###
        
        result.strip()
        
        results_list = convert_results(result, test_case["key"])
        
        output_df = pd.read_csv(testing_file)
        output_df[test_case["key"]] = [test_case["scenario"]] + results_list
        output_df.to_csv(testing_file, index=False)
    
        # Write the prompts and results to the file
        os.makedirs("prompts", exist_ok=True)
        os.makedirs("responses", exist_ok=True)
        with open(f'prompts/{model}-{test_case["key"].split("_")[0]}.txt', "a") as file:
            file.write(f'{inputs}\n====\n')
        with open(f'responses/{model}-{test_case["key"].split("_")[0]}.txt', "a") as file:
            file.write(f'{result}\n====\n')
