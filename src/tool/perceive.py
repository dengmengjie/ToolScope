from tool.base_tool import BaseTool
from typing import List, Any, Callable


Perceive_Module_PROMPT = """You are an advanced visual expert. Given an image and a question, your task is to perceive the visual content carefully and provide an accurate, concise answer. Focus on what can be directly observed or reasonably inferred from the image.

Question: 
{question}
"""


class PerceiveTool(BaseTool):
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "Perceive"
    
    @property
    def description(self) -> str:
        return """This module can percerive visual content to answer simple questions about an image. It is especially useful for answering visual sub-questions such as identifying objects, counting instances, estimating attributes (e.g., age, color, size), and describing spatial relationships. Use this module when reasoning requires detailed understanding of the image. When you need this module, state your request as: "<perceive> your question </perceive>"."""
    
    @property
    def short_description(self) -> str:
        return """When you need Perceive module, use "<perceive> your question </perceive>"."""
    
    @property
    def example(self) -> str:
        return """Question:
What is the age gap between these two people in image?

Answer: 
To find the age gap between these two people, we need to find out the age for each of them.
<perceive> What is the approximate age of the person on the left? </perceive>
<result> Around 10 years old. </result> 
Also, the person on the right is around 30 years old. Therefore, the final answer is \\boxed{{20}}."""

    def run(self, question: str, image_path: str, infer: Callable[[List[Any], bool], str]) -> str:
        text_content = Perceive_Module_PROMPT.format(question=question)
        inputs = [{
            'text_content': text_content,
            'image_path': image_path,
        }]
        output = infer(inputs, use_tqdm=False)[0]
        return output
