import json

from dataset.base import BaseDataset, normalize


demo_prompt = """
Please read the following example. Then identify which option the model chose and output its index as a single integer with no additional text. The index starts from 0.

Question: Which of these states is farthest north?
Options: ["West Virginia", "Louisiana", "Arizona", "Oklahoma"]

Model response: The state that is farthest north among the options is West Virginia.

Extracted answer: 0

Question: Which tense does the sentence use?\nMona will print her name with care.*
Options: ["present tense", "future tense", "past tense"]

Model response: The sentence uses future tense, as indicated by the word "will".

Extracted answer: 1

Question: Which of the following contains a vague pronoun reference?
Options: ["Steven's brother Jim wondered whether he ran fast enough to qualify for the Boston Marathon.", "Steven's brother Jim wondered whether Steven ran fast enough to qualify for the Boston Marathon."]

Model response: The sentence with the vague pronoun reference is the first one, because "he" is unclear.

Extracted answer: 0
"""


def extract_prediction(question, image_path, response, infer):
    test_prompt = f"\nQuestion: {question}\nModel response: {response}\nExtracted answer: "
    prompt = demo_prompt + '\n' + test_prompt
    inputs = [{"text_content": prompt,
               "image_path": image_path}]
    outputs = infer(inputs, use_tqdm=False)[0]
    return outputs


class ScienceQA(BaseDataset):
    def __init__(self, data_path="ScienceQA", split='testvqa', infer=None):
        self.data_path = data_path
        self.file_path = f"{data_path}/{split}_problems.json"
        self.infer = infer

    def load(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        # Convert the data to a list of dictionaries
        data_list = []
        for pid, item in data.items():
            item['pid'] = pid
            item['image'] = f"images/testvqa/{pid}/{item['image']}"
            question = item['question']
            context = item['hint']
            choices = item['choices']
            item['raw_question'] = item['question']
            if context == "":
                item['question'] = f"{question}\nOptions: {choices}"
            else:
                item['question'] = f"Context: {context}\nQuestion: {question}\nOptions: {choices}"
            data_list.append(item)
        return data_list

    def evaluate_results(self, results):
        total = len(results)
        correct = 0
        for sample in results:
            question = sample['item']['question']
            image_path = sample['item']['image_path']
            response = sample['generation'][-1]['output']
            answer = sample['item']['answer']
            choices = sample['item']['choices']

            prediction = extract_prediction(
                question, 
                image_path,
                response,
                self.infer
            ).strip()
            for i, choice in enumerate(choices):
                if normalize(choice) == prediction:
                    prediction = i
            if prediction.isdigit() and int(prediction) < len(choices):
                prediction = int(prediction)

            score = int(prediction == answer)
            sample['evaluation'] = {
                'prediction': prediction,
                'score': score
            }
            correct += score
        print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total:.4f}")
        return results


