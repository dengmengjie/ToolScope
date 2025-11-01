import os
import json
from dataset.base import BaseDataset, compute_f1, exact_match_score


demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Question: What color is the hydrant?

Model response: The hydrant is red.

Extracted answer: red

Question: Is the man skateboarding on a boardwalk?

Model response: Yes, the man is skateboarding on a boardwalk. You can see the wooden planks beneath him and railings along the sides, typical features of a boardwalk.

Extracted answer: yes

Question: Why are the men jumping?

Model response: The men are jumping to catch a frisbee that's in mid-air. You can see their eyes focused upward and their arms reaching out toward the flying disc.

Extracted answer: to catch a frisbee
"""


def extract_prediction(question, image_path, response, infer):
    test_prompt = f"\nQuestion: {question}\nModel response: {response}\nExtracted answer: "
    prompt = demo_prompt + '\n' + test_prompt
    inputs = [{"text_content": prompt,
               "image_path": image_path}]
    outputs = infer(inputs, use_tqdm=False)[0]
    return outputs


class VQAv2(BaseDataset):
    def __init__(self, data_path="vqav2/data", split='', infer=None):
        self.data_path = data_path
        self.file_path = os.path.join(data_path, f'data.json')
        self.infer = infer

    def load(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
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
        }

        print(f"Total: {total}, None: {count_none}")
        print(*[f"{scores[k]*100:.2f}" for k in ['avg_f1', 'avg_em']])
        return results

