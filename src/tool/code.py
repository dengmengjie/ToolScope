from tool.base_tool import BaseTool

import os
import io
import regex
import pickle
import copy
import datetime
import dateutil.relativedelta
import multiprocess
from multiprocess import Pool
from typing import Any, Dict, Optional
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []
    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime.datetime, 
        'timedelta': dateutil.relativedelta.relativedelta,
        'relativedelta': dateutil.relativedelta.relativedelta
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class CodeTool(BaseTool):
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 5,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.pool = Pool(multiprocess.cpu_count())
        self.timeout_length = timeout_length

    @property
    def name(self) -> str:
        return "Code"
    
    @property
    def description(self) -> str:
        return """This module allows you to write and execute Python code to perform tasks such as calculations, data analysis, simulations, or solving algorithmic problems. It's ideal when a task requires logic implementation and mathematical computation. When you need this module, state your Python code as: "<code> your Python code </code>". Ensure your code is complete, syntactically correct, and uses proper Python naming conventions."""
    
    @property
    def short_description(self) -> str:
        return """When you need Code module, use <code> to run Python code and end with </code>."""
    
    @property
    def example(self) -> str:
        return """Question:
What is the sum of the first 50 positive integers?

Answer: 
To find the sum of the first 50 positive integers, we can use the formula for the sum of an arithmetic series:
<code>
n = 50
sum = n * (n + 1) / 2
print(sum)
</code>
<result>1275</result> Therefore, The final answer is \\boxed{{1275}}."""

    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
    ):
        try:
            # 确保代码是字符串而不是列表
            if isinstance(code, list):
                code = '\n'.join(code)
            
            # 移除所有前导空格
            # code = '\n'.join(line.strip() for line in code.split('\n'))
            
            if get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    timeout(timeout_length)(runtime.exec_code)(code)
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
            else:
                # 分离最后一行作为表达式
                code_lines = code.split('\n')
                if len(code_lines) > 1:
                    exec_code = '\n'.join(code_lines[:-1])
                    eval_code = code_lines[-1]
                    timeout(timeout_length)(runtime.exec_code)(exec_code)
                    result = timeout(timeout_length)(runtime.eval_code)(eval_code)
                else:
                    result = timeout(timeout_length)(runtime.eval_code)(code)
                    
            report = "Done"
            str(result)
            pickle.dumps(result) # serialization check
        except Exception as e:
            result = ''
            report = str(e)
        return result, report

    def run(self, code:str) -> str:
        return self.batch_apply([code])[0][0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        with ProcessPool(max_workers=min(len(all_code_snippets), os.cpu_count())) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length, # this timeout not work
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:  
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")  
            else:  
                progress_bar = None 

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    exit()
                if progress_bar is not None:
                    progress_bar.update(1) 
            
            if progress_bar is not None:
                progress_bar.close() 

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results
