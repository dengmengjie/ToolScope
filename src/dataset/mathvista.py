import json
import os
import re
from Levenshtein import distance
import sys
sys.path.append("..")

from dataset.base import BaseDataset


demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

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
"""



def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(
    extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False
):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # if the extraction is empty, return None
        if ignore_empty_extractions and not extraction:
            return None

        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), precision))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def extract_prediction(answer_type, question, image_path, response, infer):
    test_prompt = f"Please answer the question and provide the final answer of the type {answer_type} at the end.\nQuestion: {question}\nModel response: {response}\nExtracted answer: "
    prompt = demo_prompt + '\n' + test_prompt
    inputs = [{"text_content": prompt,
               "image_path": image_path}]
    outputs = infer(inputs, use_tqdm=False)[0]
    return outputs

    
class MathVista(BaseDataset):
    def __init__(self, data_path="MathVista", split='testmini', subset_num=-1, infer=None):
        self.data_path = data_path
        self.split = split
        self.subset_num = subset_num
        self.infer = infer

        self.file_path = os.path.join(data_path, f'{split}.json')

    def load(self):
        # read data
        with open(self.file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Convert the data to a list of dictionaries
        data_list = []
        for pid, item in data.items():
            item['pid'] = pid
            raw_question = item['question']
            choices = item['choices']
            answer_type = item['answer_type']
            if choices:
                question = f"{raw_question}\nOptions: {choices}"
            else:
                question = f"{raw_question}\nThe answer shoube be {answer_type} without any units."
            item['raw_question'] = question
            item['question'] = question
            data_list.append(item)

        if self.subset_num != -1:
            data_list = data_list[:self.subset_num]
        return data_list

    def evaluate_results(self, results):
        for sample in results:
            item = sample['item']
            response = sample['generation'][-1]['output']
            answer = sample['item']['answer']

            choices = item['choices']
            question_type = item['question_type']
            answer_type = item['answer_type']
            precision = item['precision']

            response_answer = extract_prediction(
                answer_type,
                item['raw_question'],
                item['image_path'],
                response,
                infer=self.infer,
            )

            # normalize the extracted answer to match the answer type
            prediction = normalize_extracted_answer(
                response_answer,
                choices,
                question_type,
                answer_type,
                precision,
            )

            # verify the prediction is true or false
            true_false = safe_equal(prediction, answer)

            # update results
            sample['evaluation'] = {
                "prediction": prediction,
                "score": int(true_false),
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
                categories = sample['item']['metadata'][key]
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
        print(f'AVG\t{scores["average"]["accuracy"] * 100:.2f}')

        print(f"ZH\t{scores['chinese']['accuracy'] * 100:.2f}")
        print(f"EN\t{scores['english']['accuracy'] * 100:.2f}")

        print(f"FQA\t{scores['figure question answering']['accuracy'] * 100:.2f}")
        print(f"GPS\t{scores['geometry problem solving']['accuracy'] * 100:.2f}")
        print(f"MWP\t{scores['math word problem']['accuracy'] * 100:.2f}")
        print(f"TQA\t{scores['textbook question answering']['accuracy']* 100:.2f}")
        print(f"VQA\t{scores['visual question answering']['accuracy']* 100:.2f}")

        print(f"ALG\t{scores['algebraic reasoning']['accuracy'] * 100:.2f}")
        print(f"ARI\t{scores['arithmetic reasoning']['accuracy'] * 100:.2f}")
        print(f"GEO\t{scores['geometry reasoning']['accuracy'] * 100:.2f}")
        print(f"LOG\t{scores['logical reasoning']['accuracy'] * 100:.2f}")
        print(f"NUM\t{scores['numeric commonsense']['accuracy'] * 100:.2f}")
        print(f"SCI\t{scores['scientific reasoning']['accuracy'] * 100:.2f}")
        print(f"STA\t{scores['statistical reasoning']['accuracy'] * 100:.2f}")
        return results

    