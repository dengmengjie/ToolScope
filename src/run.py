import os
import json
import re
from typing import Optional
import argparse

from model.QwenVL import QwenVL
from model.InternVL import InternVL
from model.MiMoVL import MiMoVL

from utils.prompts import (
    GlobalNavigator_PROMPT,
    ToolReasoner_PROMPT,
    AnswerSummarizer_PROMPT,
)

from tool.search import SearchTool
from tool.code import CodeTool
from tool.perceive import PerceiveTool

from dataset.mathvista import MathVista
from dataset.mat import MAT
from dataset.vqav2 import VQAv2
from dataset.scienceqa import ScienceQA

# tokenizers disable parallelism to avoid deadlocks, but code executor will need multiprocessing
# https://github.com/huggingface/transformers/issues/5486#issuecomment-654232343
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


"""
data structure:
[
    {
        "item": {
            "id": id,
            "question": question,
            "image": image,
            "answer": answer
        },
        "generation": [
            {
                "text_content": text_content,
                "image_path": image_path,
                "output": output
            }
        ],
        "tool": [
            {
                "name": name,
                "result": result
            }
        ],
        "evaluation": {
            "prediction": prediction,
            "score": score,
        }
    }
]
"""


# Define special tokens
BEGIN_SEARCH = "<search>"
END_SEARCH = "</search>"

BEGIN_CODE = "<code>"
END_CODE = "</code>"

BEGIN_PERCEIVE = "<perceive>"
END_PERCEIVE = "</perceive>"

BEGIN_RESULT = "<result>"
END_RESULT = "</result>"


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset and split configuration
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path to all data."
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['MathVista', 'MAT', 'VQAv2', 'ScienceQA'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        default="",
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        "--retriever_cache_path",
        type=str,
        default=None,
        help="Path to the cache of retriever and corpus."
    )

    parser.add_argument(
        "--mm_retriever_path",
        type=str,
        default=None,
        help="path to the multimodal retriever model"   
    )

    parser.add_argument(
        "--text_corpus_path",
        type=str,
        default=None,
        help="path to the text corpus for cross-modal retrieval"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Name of the run."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=10,
        help="Maximum number of turns."
    )

    return parser.parse_args()

# extract text before the last tag
def extract_before(s: str, tag: str) -> str:
    last_marker_index = s.rfind(tag)
    
    if last_marker_index == -1:
        return s  # 如果没有找到标记，返回整个字符串
    else:
        return s[:last_marker_index]


# extract text between two tags
def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


# extract text after the first tag
def extract_after(text: str, tag: str) -> Optional[str]:
    pattern = re.escape(tag) + r"(.*)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def main():
    # ---------------------- Load Parameters ----------------------
    args = parse_args()

    data_dir = args.data_dir
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    model_path = args.model_path
    max_turn = args.max_turn
    top_k = args.top_k
    retriever_cache_path = args.retriever_cache_path
    mm_retriever_path = args.mm_retriever_path
    text_corpus_path = args.text_corpus_path
    run_name = args.run_name


    # ---------------------- Load Model ----------------------
    print('-----------------------')
    print(f'Loading model from {model_path}')
    print('-----------------------')
    # Define output directory based on model and dataset
    model_short_name = model_path.split('/')[-1]
    output_dir = f'outputs/{dataset_name}.{model_short_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model
    stop_tokens = [END_SEARCH, END_CODE, END_PERCEIVE]
    if "Qwen" in model_short_name:
        model = QwenVL(model_path=model_path, stop_tokens=stop_tokens)
    elif "Intern" in model_short_name:
        model = InternVL(model_path=model_path, stop_tokens=stop_tokens)
    elif 'MiMo' in model_short_name:
        model = MiMoVL(model_path=model_path, stop_tokens=stop_tokens)
    else:
        raise ValueError(f"Unsupported model type: {model_path}")
    infer = model.infer


    # ---------------------- Load Data ----------------------
    print('-----------------------')
    print(f'Loading data from {dataset_name} {split}')
    print('-----------------------')
    if dataset_name == 'MathVista':
        dataset = MathVista(data_path=data_dir, split=split, subset_num=subset_num, infer=infer)
    elif dataset_name == 'MAT':
        dataset = MAT(data_path=data_dir, split=split, infer=infer)
    elif dataset_name == 'VQAv2':
        dataset = VQAv2(data_path=data_dir, infer=infer)
    elif dataset_name == 'ScienceQA':
        dataset = ScienceQA(data_path=data_dir, split=split, infer=infer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    data = dataset.load()


    # ---------------------- Global Navigator ----------------------
    print("-----------------------")
    print("Running Global Navigator")
    print("-----------------------")
    # preprare sequences
    sequences = []
    for item in data:
        question = item['question']
        image_path = os.path.join(data_dir, item['image'])
        item['image_path'] = image_path

        text_content = GlobalNavigator_PROMPT.format(question=question)

        sequences.append({
            "item": item,
            "generation": [{
                'text_content': text_content,
                'image_path': image_path,
                }],
            "tool": [],
            "finished": False,
        })
    # run generation
    inputs = [seq['generation'][-1] for seq in sequences]
    outputs = infer(inputs)

    # Process outputs
    for seq, out in zip(sequences, outputs):
        match = re.search(r'\{.*?\}', out, flags=re.S)
        try:
            parsed_out = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            parsed_out = {
                "selected_tools": ["Search", "Perceive", "Code"],
                "global_plan": out
            }
            # raise ValueError("Malformed JSON inside the object.") from e
        seq['generation'][-1]['output'] = parsed_out

    output_file = os.path.join(output_dir, f'{run_name}.{split}.output.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, ensure_ascii=False, indent=4)
    print(f"Outputs saved to {output_file}")


    # ---------------------- Load Tools -----------------
    print('-----------------------')
    print("Loading tools")
    print('-----------------------')
    tool_dict = {}
    print("🔍 Loading Search Tool")
    search_tool = SearchTool(
        retriever_cache_path=retriever_cache_path,
        mm_retriever_path = mm_retriever_path,
        text_corpus_path = text_corpus_path,
    )
    tool_dict['Search'] = search_tool
    print("💻 Loading Code Tool")
    code_tool = CodeTool(get_answer_from_stdout=True)
    tool_dict['Code'] = code_tool
    print("👁️ Loading Perceive Tool")
    perceive_tool = PerceiveTool()
    tool_dict['Perceive'] = perceive_tool
    

    # ---------------------- Run generation ----------------------
    print("-----------------------")
    print("Running Tool Reasoner")
    print("-----------------------")
    # preprare initial inputs
    for seq in sequences:
        question = seq['item']['question']
        image_path = seq['item']['image_path']
        selected_tools = seq['generation'][-1]['output']['selected_tools']
        global_plan = seq['generation'][-1]['output']['global_plan']

        tool_descriptions = "\n".join([
            f"{i+1}. {tool_name}: {tool_dict[tool_name].description}" 
            for i, tool_name in enumerate(selected_tools)
        ])
        tool_short_descriptions = " ".join([
            tool_dict[tool_name].short_description 
            for tool_name in selected_tools
        ])
        tool_examples = "\n\n\n".join([
            tool_dict[tool_name].example 
            for tool_name in selected_tools
        ])

        text_content = ToolReasoner_PROMPT.format(
            tool_descriptions=tool_descriptions,
            tool_short_descriptions=tool_short_descriptions,
            tool_examples=tool_examples,
            question=question,
            previous_reasoning=global_plan,
        )

        seq['generation'].append({
            'text_content': text_content,
            'image_path': image_path,
        })
 

    # Main loop until all sequences are finished or maximum turns reached
    turn = 0
    while True:
        # Identify sequences that need generation
        unfinished_sequences = [seq for seq in sequences if not seq['finished']]

        if turn >= max_turn:
            print(f"Maximum number of turns ({max_turn}) reached, stopping.")
            break
        elif unfinished_sequences:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"We have {len(unfinished_sequences)} sequences needing generation...")

            # Generation
            inputs = [seq['generation'][-1] for seq in unfinished_sequences]
            outputs = infer(inputs)
            print("Generation completed, processing outputs...")

            # Process outputs
            search_sequences = []
            code_sequences = []
            perceive_sequences = []
            for seq, out in zip(unfinished_sequences, outputs):
                seq['generation'][-1]['output'] = out

                if out.rstrip().endswith(END_SEARCH):
                    search_sequences.append(seq)
                elif out.rstrip().endswith(END_CODE):
                    code_sequences.append(seq)
                elif out.rstrip().endswith(END_PERCEIVE):
                    perceive_sequences.append(seq)
                else:
                    seq['finished'] = True
            print(f"{len(search_sequences)} need Search Module")
            print(f"{len(code_sequences)} need Code Module")
            print(f"{len(perceive_sequences)} need Perceive Module")

            # Search: Retrieve and refine
            for seq in search_sequences:
                # preprare
                text_content = seq['generation'][-1]['text_content']
                image_path = seq['item']['image_path']
                out = seq['generation'][-1]['output']
                previous_reasoning = extract_before(out, BEGIN_SEARCH)
                calling = extract_between(out, BEGIN_SEARCH, END_SEARCH)

                # retrieve and refine
                if calling:
                    result = search_tool.run(
                        query=calling,
                        top_k=top_k,
                        question=seq['item']['question'],
                        previous_reasoning=previous_reasoning,
                        image_path=image_path,
                        infer=infer
                    )
                else:
                    calling = None
                    result = None

                # update
                seq['tool'].append({
                    "name": "Search",
                    "input": calling,
                    "output": result,
                })

                # rollback or forward
                if result is None or "No helpful information found." in result:
                    seq['generation'].append({
                        "text_content": f"{text_content}{previous_reasoning}\n",
                        "image_path": image_path,
                    })
                else:
                    seq['generation'].append({
                        "text_content": f"{text_content}{out}\n{BEGIN_RESULT}{result}{END_RESULT}",
                        "image_path": image_path,
                    })

            # Code Execution
            for seq in code_sequences:
                # preprare
                text_content = seq['generation'][-1]['text_content']
                image_path = seq['item']['image_path']
                out = seq['generation'][-1]['output']
                previous_reasoning = extract_before(out, BEGIN_CODE)
                calling = extract_between(out, BEGIN_CODE, END_CODE)

                # get result
                result = code_tool.run(calling)

                # update
                seq['tool'].append({
                    "name": "Code",
                    "input": calling,
                    "output": result,
                })

                # rollback or forward
                if result is None or result == "":
                    seq['generation'].append({
                        "text_content": f"{text_content}{previous_reasoning}\n",
                        "image_path": image_path,
                    })
                else:
                    seq['generation'].append({
                        "text_content": f"{text_content}{out}\n{BEGIN_RESULT}{result}{END_RESULT}",
                        "image_path": image_path,
                    })

            # Perceive
            for seq in perceive_sequences:
                # preprare
                text_content = seq['generation'][-1]['text_content']
                image_path = seq['item']['image_path']
                out = seq['generation'][-1]['output']
                previous_reasoning = extract_before(out, BEGIN_PERCEIVE)
                calling = extract_between(out, BEGIN_PERCEIVE, END_PERCEIVE)

                # retrieve and refine
                if calling:
                    result = perceive_tool.run(calling, image_path, infer)
                else:
                    calling = None
                    result = None

                # update
                seq['tool'].append({
                    "name": "Perceive",
                    "input": calling,
                    "output": result,
                })

                # rollback or forward
                if result is None:
                    seq['generation'].append({
                        "text_content": f"{text_content}{previous_reasoning}\n",
                        "image_path": image_path,
                    })
                else:
                    seq['generation'].append({
                        "text_content": f"{text_content}{out}\n{BEGIN_RESULT}{result}{END_RESULT}",
                        "image_path": image_path,
                    })
        else:
            print("All sequences finished.")
            break

    # ---------------------- Answer Summarizer ----------------------
    for seq in sequences:
        if 'output' not in seq['generation'][-1].keys():
            seq['generation'] = seq['generation'][:-1]

        question = seq['item']['question']
        reasoning = seq['generation'][-1]['output']
        text_content = AnswerSummarizer_PROMPT.format(question=question, reasoning=reasoning)
        image_path = seq['generation'][0]['image_path']

        seq['generation'].append({
            "text_content": text_content,
            "image_path": image_path,
        })
    
    inputs = [seq['generation'][-1] for seq in sequences]
    outputs = infer(inputs)

    for seq, out in zip(sequences, outputs):
        seq['generation'][-1]['output'] = out

    
    # ----------------------- Save Outputs ----------------------
    output_file = os.path.join(output_dir, f'{run_name}.{split}.output.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, ensure_ascii=False, indent=4)
    print(f"Outputs saved to {output_file}")


    # ---------------------- Evaluation ----------------------
    dataset.evaluate(output_file)

    print("Process completed.")

if __name__ == "__main__":
    main()
