import os
import json
from dataset.base import BaseDataset, compute_f1, exact_match_score


demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Question: Is the country where the person in the image was born the same as the country where the founder of uMkhonto we Sizwe is from?

Model response: The person in the image is Elon Musk. Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa. Nelson Mandela co-founded uMkhonto we Sizwe in 1961. He was born in Mvezo, South Africa. Yes, they were born in the same country — South Africa.

Extracted answer: Yes

Question: Who was the first president of the university from which the chief executive officer of the company in the image graduated?

Model response: The image shows the logo of Tesla. The next step is to confirm the chief executive officer of Tesla. I have confirmed that Elon Musk is the CEO of Tesla. Now, I need to find out which university Elon Musk graduated from. Elon Musk graduated from the University of Pennsylvania in the U.S. Benjamin Franklin was the founder and first president of the University of Pennsylvania. 

Extracted answer: Benjamin Franklin

Question: How old was the director of this movie when the 27th Olympic Games were held?

Model response: He was 30 years old then.

Extracted answer: 30
"""


def extract_prediction(question, image_path, response, infer):
    test_prompt = f"\nQuestion: {question}\nModel response: {response}\nExtracted answer: "
    prompt = demo_prompt + '\n' + test_prompt
    inputs = [{"text_content": prompt,
               "image_path": image_path}]
    outputs = infer(inputs, use_tqdm=False)[0]
    return outputs


class MAT(BaseDataset):
    def __init__(self, data_path="laolao77/MAT/MAT-Benchmark", split='MAT-Search', infer=None):
        self.data_path = data_path
        self.file_path = os.path.join(data_path, f'{split}.json')
        self.infer = infer

    def load(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        for item in data:
            item['image'] = os.path.join('MAT-Search-image', item['image_path'])
        return data
    
    def evaluate_results(self, results):
        total = len(results)
        count_none = 0
        f1_total = []
        em_total = []

        for sample in results:
            question = sample['item']['question']
            image_path = sample['item']['image_path']
            response = sample['generation'][-1]['output']
            answer = sample['item']['answer']
            pred = extract_prediction(
                question, 
                image_path,
                response,
                self.infer
            )

            # 若gt是str，统一转换为列表处理
            if isinstance(answer, str):
                answer = [answer]

            f1 = max([compute_f1(pred, gt) for gt in answer])
            em = max([exact_match_score(pred, gt) for gt in answer])
            if em == 1:
                f1 = 1

            f1_total.append(f1)
            em_total.append(em)
            sample['evaluation'] = {
                'prediction': pred,
                'f1_score': f1,
                'em_score': em,
            }

        scores = {
            "avg_f1": sum(f1_total) / total if total > 0 else 0,
            "avg_em": sum(em_total) / total if total > 0 else 0,
            "simple_f1": sum(f1_total[:75]) / 75 if total > 0 else 0,
            "simple_em": sum(em_total[:75]) / 75 if total > 0 else 0,
            "hard_f1": sum(f1_total[75:]) / 75 if total > 0 else 0,
            "hard_em": sum(em_total[75:]) / 75 if total > 0 else 0,
        }

        print(f"Total: {total}, None: {count_none}")
        print(*[f"{scores[k]*100:.2f}" for k in ['simple_f1', 'simple_em', 'hard_f1', 'hard_em', 'avg_f1', 'avg_em']])
        return results
