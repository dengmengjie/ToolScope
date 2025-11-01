from vllm import LLM, SamplingParams
import torch
from typing import List
import PIL


class InternVL:

    def __init__(self, model_path: str, stop_tokens: List[str] = []):
        self.model_path = model_path

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.85,
            limit_mm_per_prompt={"image": 10},
            trust_remote_code=True, 
            # max_model_len=13200,
            # enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            # seed=seed,
            max_tokens=8192,
            repetition_penalty=1.05,
            top_k=40, 
            top_p=0.8, 
            temperature=0.8,
            stop=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"] + stop_tokens,
            include_stop_str_in_output=True,
        )
        

    def infer(self, inputs, use_tqdm: bool = True):
        # prepare inputs
        llm_inputs = []
        for inp in inputs:
            text_content = inp['text_content']
            image_path = inp['image_path']
            image_content = PIL.Image.open(image_path)

            text_content = f"USER: <image>\n{text_content}\nASSISTANT:"

            inputs = {
                "prompt": text_content,
                "multi_modal_data": {"image": image_content},
            }

            llm_inputs.append(inputs)

        # run generation
        llm_outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params, use_tqdm=use_tqdm)
        outputs = [out.outputs[0].text for out in llm_outputs]
        return outputs
