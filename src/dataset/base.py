# define a base class for dataset
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import json
import re
import string


def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    if prediction is None:
        return 0.0
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))


def extract_prediction_in_tag(response):
    # extract prediction between <answer> and </answer>
    matches = re.findall(r"<answer>(.*?)</answer>", response)
    if matches:
        prediction = matches[-1]
    else:
        prediction = response.strip()
    return prediction


class BaseDataset(ABC):
    """
    Base class for datasets.
    load data, evaluate method.
    """

    def __init__(self, data_path):
        """
        Initialize the dataset with a data path.
        :param data_path: Path to the dataset file or directory.
        """
        self.data_path = data_path
        self.load()

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Load the dataset.
        Returns a list of dictionaries where each dictionary represents a data point.
        """
        pass

    @abstractmethod
    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Union[str, float]]:
        """
        Evaluate the predictions against the ground truth.
        :param predictions: List of dictionaries containing predictions.
        :return: A dictionary with evaluation metrics.
        """
        pass

    def evaluate(self, output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        results = self.evaluate_results(results)

        print(f"Evaluation results saved to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            data_json = json.dumps(results, indent=4)
            f.write(data_json)