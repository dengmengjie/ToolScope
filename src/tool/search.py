from tool.base_tool import BaseTool
from typing import List, Any, Callable
import bm25s
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# Example = """Question: 
# What is the perimeter of the rectangle?
# Choices:
# (A) 8
# (B) 12
# (C) 16
# (D) 20

# Answer:
# From the image, we can see that the rectangle has dimensions: Length (l) = 5 ft, Width (w) = 3 ft. To find the perimeter of a rectangle, we need to search for the formula. <|begin_search|> formula of perimeter of a rectangle with given length and width. <|end_search|> <|begin_result|>The formula for the perimeter of a rectangle with a given length (L) and width (W) is: Perimeter = 2 \\times (L + W).<|end_result|> Therefore, the perimeter can be calculated as: 2 \\times (5 + 3) = 16 ft. The final answer is \\boxed{{C}}."""


Example = """Question: 
Find $x$ so that each quadrilateral is a parallelogram.

Answer:
The image shows a quadrilateral that resembles a parallelogram. The left side is 2x-5, and the right side is 3x-18. To determine the value of x that makes the quadrilateral a parallelogram, we need to use the property of a parallelogram to create an equation. <search> Properties of parallelograms </search> <result> In a parallelogram, opposite sides are equal. </result> Therefore, 2x-5=3x-18. So, x=13. The final answer is \\boxed{{13}}."""


Integrate_PROMPT = """You are an advanced reasoning agent. Given the Question about an image, the Previous Reasoning Steps, a Current Search Query, and a set of Searched Documents, your task is to:
1. Analyze the Previous Reasoning Steps and Current Search Query to understand the assistant's current objective and what specific information is required.
2. Read Searched Documents and identify content directly relevant to the Current Search Query.
3. Integrate and rephrase the extracted information smoothly into the reasoning chain. Do not quote or copy verbatim. Instead, answer the Current Search Query using fluent, natural language. Phrase the information as internal knowledge or informed commentary, using expressions like "According to external sources", "As is known", or "As the search results show".


Here is an example:
Question:
Find $x$ so that each quadrilateral is a parallelogram.

Previous Reasoning Steps:
The image shows a quadrilateral that resembles a parallelogram. The left side is 2x-5, and the right side is 3x-18. To determine the value of x that makes the quadrilateral a parallelogram, we need to use the property of a parallelogram to create an equation.

Current Search Query:
Properties of parallelograms

Searched Documents:
Passage 1: 
A simple (non-self-intersecting) quadrilateral is a parallelogram if and only if any one of the following statements is true:
Two pairs of opposite sides are parallel (by definition).
Two pairs of opposite sides are equal in length.
Two pairs of opposite angles are equal in measure.
The diagonals bisect each other.
One pair of opposite sides is parallel and equal in length.
Adjacent angles are supplementary.

Output:
In a parallelogram, opposite sides are equal.


Now complete the task for the input below:
Question:
{question}

Previous Reasoning Steps:
{previous_reasoning}

Current Search Query:
{calling}

Searched Documents:
{raw_result}

Output:
"""


class SearchTool(BaseTool):
    def __init__(self, retriever_cache_path: str, mm_retriever_path: str, text_corpus_path: str):
        super().__init__()
        print("Loading bm25 model and corpus from cache")
        self.text_retriever = bm25s.BM25.load(retriever_cache_path, load_corpus=True)

        self.mm_model = CLIPModel.from_pretrained(mm_retriever_path)
        self.mm_processor = CLIPProcessor.from_pretrained(mm_retriever_path)

        with open(text_corpus_path, "r", encoding="utf-8") as f:
            self.texts = []
        inputs = self.mm_processor(text=self.texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.mm_model.get_text_features(**inputs)
            self.text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        return

    @property
    def name(self) -> str:
        return "Search"

    @property
    def description(self) -> str:
        return """This module performs searches on Wikipedia to gather relevant information for any query. It's especially useful for retrieving knowledge, definitions and problem-solving strategies. When you need this module, state your request as: "<search> your query or topic of interest </search>". If you want to use the image as query, state your request as "<search> image </search>". """
    
    @property
    def short_description(self) -> str:
        return """When you need Search module, use <search> to request a search and end with </search>."""
    
    @property
    def example(self) -> str:
        return Example
    
    def text_retrieve(self, query: str, top_k: int=1) -> str:
        # tokenize query
        query_tokens = bm25s.tokenize(query)

        # retrieve
        results = self.retriever.retrieve(query_tokens, k=top_k, return_as="documents")[0]
        knowledge_list = [x['text'] for x in results]
        raw_result = "\n".join([f"Passage {i+1}: {psg}" for i, psg in enumerate(knowledge_list)])
        return raw_result

    def mm_retrieve(self, image_path: str, top_k: int=1) -> str:
        image = Image.open(image_path)
        image_inputs = self.mm_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_embedding = self.mm_model.get_image_features(**image_inputs)   # shape: [1, 512]
            image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)

        similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)  # shape: [N]
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        raw_results = [self.texts[top_indices[i]] for i in range(top_k) if top_scores[i] > 0.9]
        return raw_results

    def run(self, query: str, top_k: int, question: str, previous_reasoning: str, image_path: str, infer: Callable[[List[Any], bool], str]) -> str:
        if query.strip() == "image":
            raw_result = self.mm_retrieve(image_path, top_k)
        else:
            raw_result = self.text_retrieve(query, top_k)

        # refine
        text_content = Integrate_PROMPT.format(question=question,
                                               previous_reasoning=previous_reasoning, 
                                               calling=query, 
                                               raw_result=raw_result)
        inputs = [{
            'text_content': text_content,
            'image_path': image_path,
        }]
        result = infer(inputs, use_tqdm=False)[0]
        return result
