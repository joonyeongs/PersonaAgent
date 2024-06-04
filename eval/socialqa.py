import sys
import os
import argparse
import json
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_path)
sys.path.append(parent_dir)

from models.model import load_model, model_generate


def socialqa(args):
    with open('/home/vqa/data/dataset/socialiqa-train-dev/dev-labels.lst', 'r') as file:
        lines = file.readlines()
        labels = [chr(int(line.strip()) + ord('A') - 1) for line in lines]

    tokenizer, model = load_model(args.model_dir)
    system = f"""
    You are an {args.mbti}. Engage in daily conversations with the user, providing friendly and responsive dialogue. 
    Be attentive and offer thoughtful responses to any topic the user wishes to discuss. 
    Respond the dialogue based on your preferences and personal traits.
    """

    system = "You are a helpful AI assistant."

    with open(args.input_dir, "r") as f:
        dataset = [json.loads(line) for line in f]
    outputs = []

    for i, data in enumerate(dataset):
        try:
            context = data["context"]
            question = data["question"]
            answerA = data["answerA"]
            answerB = data["answerB"]
            answerC = data["answerC"]

            prompt = f"{context} {question} A: {answerA} B: {answerB} C: {answerC}\nYou have to choose the best answer. Answer in the format: 'The best answer is: '. Also, provide a reason for your choice."
            output = model_generate(system, prompt, tokenizer, model)
            outputs.append({
                "context": context,
                "question": question,
                "answerA": answerA,
                "answerB": answerB,
                "answerC": answerC,
                "response": output,
                "label": labels[i],
            })
            logging.info(f'Processed dataset item {i} successfully for {args.mbti}.')
        except Exception as e:
            logging.error(f'Error processing dataset item {i}: {e}')

    output_dir = f"{args.output_dir}/{args.mbti}.json"
    with open(output_dir, "w") as f:
        json.dump(outputs, f, indent=4)

    

def eval_socialqa(args):
    output_dir = f"{args.output_dir}/{args.mbti}.json"

    with open(output_dir, "r") as f:
        outputs = json.load(f)

    correct_count = 0
    total_count = len(outputs)
    outputs_eval = []
    for i, output in enumerate(outputs):
        response = output["response"]
        label = output["label"]
        if "The best answer is: " in response:
            chosen_answer = response.split("The best answer is: ")[1].strip()[0]
            if chosen_answer == label:
                correct_count += 1
        else:
            print(f"Error: No answer chosen for dataset item {i}")
            chosen_answer = None

        outputs_eval.append({
            "context": output["context"],
            "question": output["question"],
            "answerA": output["answerA"],
            "answerB": output["answerB"],
            "answerC": output["answerC"],
            "response": response,
            "label": label,
            "chosen_answer": chosen_answer,
        })
    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy}")

    output_dir = f"{args.output_dir}/{args.mbti}_eval.json"
    with open(output_dir, "w") as f:
        json.dump(outputs_eval, f, indent=4)

def eval_socialqa_accuracy(args):
    output_dir = f"{args.output_dir}/{args.mbti}_eval.json"

    with open(output_dir, "r") as f:
        outputs = json.load(f)

    correct_count = 0
    total_count = len(outputs)
    for output in outputs:
        label = output["label"]
        chosen_answer = output["chosen_answer"]
        if chosen_answer == label:
            correct_count += 1

    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbti", type=str, help="Which MBTI type to use")
    parser.add_argument("--model-dir", type=str, help="Directory of the model")
    parser.add_argument("--input-dir", type=str, help="Input directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    #socialqa(args)
    eval_socialqa(args)    