import argparse
import re
import json
from tqdm import tqdm
import sys
sys.path.append("..")

import random
random.seed(a=42, version=2)


extract_prediction_prompt = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B

Hint: Please answer the question and give the final answer at the end.
Question: {question}

Model response: {response}

Extracted answer: """


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        data_json = json.dumps(data, indent=4)
        f.write(data_json)


def extract_prediction(response, item):
    # extract text in \\boxed{}
    matches = re.findall(r"\\boxed\{(.*?)\}", response)
    if matches:
        prediction = matches[-1]
    else:
        prediction = ""
    # extract text in \\text{}
    matches = re.findall(r"\\text\{(.*?)\}", prediction)
    if matches:
        prediction = matches[-1]
    # extract text in ()
    matches = re.findall(r"\((.*?)\)", prediction)
    if matches:
        prediction = matches[-1]
    
    # if prediction is capital choice, transform to real value choice
    choices = item["choices"]
    if choices:
        capital_choices = [chr(65 + i) for i in range(len(choices))]
        if prediction in capital_choices:
            prediction = choices[ord(prediction) - 65]
    return prediction


# def extract_prediction(response, item, infer):
#     question = item['question']
#     image_path = item['image_path']
#     text_content = extract_prediction_prompt.format(question=question, response=response)
#     inputs = [{
#         'text_content': text_content,
#         'image_path': image_path,
#     }]
#     prediction = infer(inputs, use_tqdm=False)
#     prediction = prediction[0].strip()
#     return prediction


def evaluate_results(results, infer):
    # score each sample
    for sample in tqdm(results):
        item = sample['item']
        if 'output' in sample['generation'][-1]:
            response = sample['generation'][-1]['output']
        else:
            response = sample['generation'][-2]['output']
        answer = sample['item']['answer']
        
        # extract prediction
        prediction = extract_prediction(response, item)
        score = 1 if prediction == answer else 0
        sample['evaluation'] = {
            'prediction': prediction,
            'score': score
        }
    
    # count and print tables
    total = len(results)
    correct = sum([sample['evaluation']['score'] for sample in results])
    accuracy = correct / total
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    # count by fine-grained categories
    target_keys = [
        'language',
        'task',
        'skills',
    ]
    for sample in results:
        for key in target_keys:
            categories = sample['item'][key]
            # check if category is a list
            if not isinstance(categories, list):
                categories = [categories]
            for category in categories:
                if category not in scores.keys():
                    scores[category] = {
                        'total': 1,
                        'correct': sample['evaluation']['score']
                    }
                else:
                    scores[category]['total'] += 1
                    scores[category]['correct'] += sample['evaluation']['score']
    for category in scores.keys():
        scores[category]['accuracy'] = scores[category]['correct'] / scores[category]['total']
    # sort key_dict by category
    scores = dict(sorted(scores.items(), key=lambda item: float(item[1]['accuracy']), reverse=True))
    print(scores.keys())

    # print results
    print(f'AVG\t{scores["average"]["accuracy"] * 100:.1f}')

    print(f"ZH\t{scores['chinese']['accuracy'] * 100:.1f}")
    print(f"EN\t{scores['english']['accuracy'] * 100:.1f}")

    print(f"FQA\t{scores['figure question answering']['accuracy'] * 100:.1f}")
    print(f"GPS\t{scores['geometry problem solving']['accuracy'] * 100:.1f}")
    print(f"MWP\t{scores['math word problem']['accuracy'] * 100:.1f}")
    print(f"TQA\t{scores['textbook question answering']['accuracy']* 100:.1f}")
    print(f"VQA\t{scores['visual question answering']['accuracy']* 100:.1f}")

    print(f"ALG\t{scores['algebraic reasoning']['accuracy'] * 100:.1f}")
    print(f"ARI\t{scores['arithmetic reasoning']['accuracy'] * 100:.1f}")
    print(f"GEO\t{scores['geometry reasoning']['accuracy'] * 100:.1f}")
    print(f"LOG\t{scores['logical reasoning']['accuracy'] * 100:.1f}")
    print(f"NUM\t{scores['numeric commonsense']['accuracy'] * 100:.1f}")
    print(f"SCI\t{scores['scientific reasoning']['accuracy'] * 100:.1f}")
    print(f"STA\t{scores['statistical reasoning']['accuracy'] * 100:.1f}")

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('-f', '--results_file_path', type=str, default=None)
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # # load model
    # model_path = args.model_path
    # model_short_name = model_path.split('/')[-1]
    # if "Qwen" in model_short_name:
    #     model = QwenVL(model_path=model_path)
    # elif "InternVL" in model_short_name:
    #     model = InternVL(model_path=model_path)
    # else:
    #     raise ValueError(f"Unsupported model type: {model_path}")
    # infer = model.infer

    # load results
    with open(args.results_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    output_path = args.results_file_path
    
    results = evaluate_results(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        data_json = json.dumps(results, indent=4)
        f.write(data_json)
