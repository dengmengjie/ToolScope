from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
from typing import List
import PIL


class MiMoVL:

    def __init__(self, model_path: str, stop_tokens: List[str] = []):
        self.model_path = model_path

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            tokenizer_kwargs={"use_fast": True}
        )
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.85,
            limit_mm_per_prompt={"image": 10},
        )
        self.sampling_params = SamplingParams(
            # seed=seed,
            max_tokens=32768,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            stop=[self.processor.tokenizer.eos_token] + stop_tokens,
            include_stop_str_in_output=True,
        )

    def infer(self, inputs, use_tqdm: bool = True):
        # prepare inputs
        llm_inputs = []
        for inp in inputs:
            text_content = inp['text_content']
            image_path = inp['image_path']
            image_content = PIL.Image.open(image_path)
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image",
                            "image": image_content,
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {
                            "type": "text",
                            "text": text_content
                        },
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            }

            llm_inputs.append(inputs)

        # run generation
        llm_outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params, use_tqdm=use_tqdm)
        outputs = [out.outputs[0].text for out in llm_outputs]
        return outputs
