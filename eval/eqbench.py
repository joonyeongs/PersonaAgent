import sys
import os
import argparse
import json
import logging
import re
import math
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_path)
sys.path.append(parent_dir)

from models.model import load_model, model_generate

    

def eqbench(args):
    tokenizer, model = load_model(args.model_dir)
    system = f"""
    You are an {args.mbti}. Engage in daily conversations with the user, providing friendly and responsive dialogue. 
    Be attentive and offer thoughtful responses to any topic the user wishes to discuss. 
    Respond the dialogue based on your preferences and personal traits.
    """

    if args.mbti == "none":
        system = "You are a helpful AI assistant."

    with open(args.input_dir, "r") as f:
        #dataset = [json.loads(line) for line in f]
        dataset = json.load(f)
    outputs = []

    for i, data in enumerate(dataset):
        try:
            prompt = data['prompt']

            output = model_generate(system, prompt, tokenizer, model)
            outputs.append({
                **data,
                "response": output,
            })
            logging.info(f'Processed dataset item {i} successfully for {args.mbti}.')
        except Exception as e:
            logging.error(f'Error processing dataset item {i}: {e}')

    output_dir = f"{args.output_dir}/{args.mbti}.json"
    with open(output_dir, "w") as f:
        json.dump(outputs, f, indent=4)

    
def parse_answers(text, revise=True):
    first_pass_answers = {}
    revised_answers = {}

	# Extracting first pass answers
    if revise:
        first_pass_match = re.search(r'First pass scores:(.*?)Revised scores:', text, re.DOTALL)
        if first_pass_match:
            first_pass_text = first_pass_match.group(1)
            first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', first_pass_text))

        if first_pass_answers == {}:
            first_pass_match = re.search(r'First pass scores:(.*?)$', text, re.DOTALL)
            if first_pass_match:
                first_pass_text = first_pass_match.group(1)
                first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', first_pass_text))

		# Extracting revised answers
        revised_match = re.search(r'Revised scores:(.*?)$', text, re.DOTALL)
        if revised_match:
            revised_text = revised_match.group(1)
            revised_answers = dict(re.findall(r'(\w+):\s+(\d+)', revised_text))
    else:
        first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', text))
        revised_answers = {}

    return first_pass_answers, revised_answers


def is_parsed_correctly(answer):
    if len(answer) != 4:
        return False
    return True
     


def eval_eqbench(args):
    output_dir = f"{args.output_dir}/{args.mbti}.json"

    with open(output_dir, "r") as f:
        outputs = json.load(f)

    outputs_evaluated = []
    for i, output in enumerate(outputs):
        response = output["response"]
        first_pass_answer, revised_answer = parse_answers(response)
        outputs_evaluated.append({
            **output,
            "first_pass_answer": first_pass_answer,
            "revised_answer": revised_answer,
        })
        
        if not is_parsed_correctly(first_pass_answer):
            logging.info(f"Error parsing {i+1}/{len(outputs)} of {args.mbti} - first pass answers.")
        elif not is_parsed_correctly(revised_answer):
            logging.info(f"Error parsing {i+1}/{len(outputs)} of {args.mbti} - revised answers.")
        else:
            logging.info(f"Finished {i+1}/{len(outputs)} of {args.mbti}")
        
        
    output_dir = f"{args.output_dir}/{args.mbti}_eval.json"
    with open(output_dir, "w") as f:
        json.dump(outputs_evaluated, f, indent=4)


def calculate_score_fullscale(reference, user):
	# First check that the emotions specified in the answer match those in the reference
	if len(user.items()) != 4:
		#print('! Error: 4 emotions were not returned')
		#print(user)
		return None
	emotions_dict = {}
	for emotion, user_emotion_score in user.items():
		for i in range(1, 5):
			if emotion.lower() == reference[f'emotion{i}'].lower():
				emotions_dict[emotion.lower()] = True
	if len(emotions_dict) != 4:
		print('! Error: emotions did not match reference')
		print(user)
		return None
	
	difference_tally = 0  # Tally of differerence from reference answers for this question
	
	# Iterate over each emotion in the user's answers.
	for emotion, user_emotion_score in user.items():
		# If this emotion is in the reference, calculate the difference between the user's score and the reference score.
		for i in range(1, 5):
			if emotion.lower() == reference[f'emotion{i}'].lower():
				d = abs(float(user_emotion_score) - float(reference[f'emotion{i}_score']))
				# this will be a value between 0 and 10
				if d == 0:
					scaled_difference = 0
				elif d <= 5:
					# S-shaped scaling function
					# https://www.desmos.com/calculator
					# 6.5\cdot\ \frac{1}{\left(1\ +\ e^{\left(-1.2\cdot\left(x-4\right)\right)}\right)}						
					scaled_difference = 6.5 * (1 / (1 + math.e ** (-1.2 * (d-4))))

				else:
					scaled_difference = d
				difference_tally += scaled_difference
					
	# Inverting the difference tally so that the closer the answer is to reference, the higher the score.
	# The adjustment constant is chosen such that answering randomly produces a score of zero.
	adjust_const =  0.7477
	final_score = 10 - (difference_tally * adjust_const)
	
	return final_score

def eval_eqbench_accuracy(args):
    output_dir = f"{args.output_dir}/{args.mbti}_eval.json"

    with open(output_dir, "r") as f:
        outputs = json.load(f)

    outputs_evaluated = []
    first_pass_scores, revised_scores = [], []
    for i, output in enumerate(outputs):
        first_pass_answer = output["first_pass_answer"]
        revised_answer = output["revised_answer"]
        reference = output["reference_answer_fullscale"]

        first_pass_score = calculate_score_fullscale(reference, first_pass_answer)
        revised_score = calculate_score_fullscale(reference, revised_answer)
        if first_pass_score is not None:
             first_pass_scores.append(first_pass_score)
        if revised_score is not None:
            revised_scores.append(revised_score)

        outputs_evaluated.append({ 
            **output,
            "first_pass_score": first_pass_score,
            "revised_score": revised_score,
        })

        logging.info(f"First pass score: {first_pass_score} / Revised score: {revised_score} for {args.mbti} {i+1}/{len(outputs)}")

    
    first_pass_avg = sum(first_pass_scores) / len(first_pass_scores) 
    revised_avg = sum(revised_scores) / len(revised_scores)

    print(f"First pass average: {first_pass_avg * 10}")
    print(f"Revised average: {revised_avg * 10}")

    output_dir = f"{args.output_dir}/{args.mbti}_eval_accuracy.json"
    with open(output_dir, "w") as f:
        json.dump(outputs_evaluated, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbti", type=str, help="Which MBTI type to use")
    parser.add_argument("--model-dir", type=str, help="Directory of the model")
    parser.add_argument("--input-dir", type=str, help="Input directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    #eqbench(args)
    eval_eqbench_accuracy(args)    