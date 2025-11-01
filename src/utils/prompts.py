DirectPrompting_PROMPT = """Please answer the following math question. Provide your final answer in the format \\boxed{{YOUR_ANSWER}}.

Question: {input}"""


CoT_PROMPT = """Please answer the following math question. You should think step by step to solve it. Provide your final answer in the format \\boxed{{YOUR_ANSWER}}.

Question: {input}"""


# GlobalNavigator_PROMPT = """You are a high-level planner. Given a question about an image, provide a concise strategic overview to guide reasoning. Your output should:
# 1. Understand the core intent of the question.
# 2. Extract key principles or domain knowledge relevant to solving it.
# 3. Propose a high-level strategy or line of inquiry.
# 4. Optionally, suggest sub-questions or tools that might help (e.g., search, vision, code).

# Avoid detailed reasoning or answers. Focus on guidance, not execution.

# Question:
# {question}"""


GlobalNavigator_PROMPT = """You are an expert planner in a multimodal reasoning system. Your role is to perform high-level strategic analysis and select the most appropriate tools to help solve a given question about an image.

Your objectives are:
1. Tool Selection: From the available tools listed below, identify which are necessary to solve the task.
2. Global Reasoning Plan: Write a concise, high-level plan that outlines the sequence of major reasoning steps needed to arrive at the correct answer. This plan should reflect how the selected tools will be used and in what order.

Available Tools:
1. Search: Retrieve factual or background knowledge relevant to the question.
2. Code: Perform mathematical computations or logical operations.
3. Perceive: Extract fine-grained visual information (e.g., text, objects, layout) from the image.

Guidelines:
1. If the task is simple and solvable directly by the model, you may decide to use no tools.
2. For complex tasks, combine tools modularly to handle different reasoning needs.
3. Be explicit about why each tool is selected and how it contributes to the reasoning plan.

Output Format:
Return your answer in JSON format with two keys: "selected_tools" and "global_plan".
Example:
{{
    "selected_tools": ["Search", "Code"],
    "global_plan": "First, use Search to retrieve background knowledge about the scientific concept in the question. Then, extract relevant values from the image using Perceive. Finally, use Code to compute the final result based on the retrieved and extracted information."
}}

Question: 
{question}
"""



ToolReasoner_PROMPT = """You are an advanced question-answering agent equipped with specialized modules to aid in analyzing and responding to queries about images:
{tool_descriptions}


When faced with a question about an image, your task is to:
1. Reason step by step.
2. Utilize modules during your reasoning. {tool_short_descriptions}
3. Give the final answer. 


Here are some examples:
{tool_examples}


Please refer to the prompts and examples above to help me solve the following problem. 
Question: 
{question}

Answer: 
{previous_reasoning}"""


AnswerSummarizer_PROMPT = """You will be given a question, an image, and a solution generated from previous steps. Your task is to form a clear reasoning path, extract the final answer from the reasoning result and reformat it for evaluation.

Give the final answer directly. If you are given Options, your answer should be one of them. Otherwise, your answer should be very brief. 

Question: 
{question}

Answer: 
{reasoning}
"""


# AnswerSummarizer_PROMPT = """You will be given a question, an image, and a solution generated from previous steps. Your task is to extract the final answer from the reasoning result and reformat it for evaluation.

# Give the final answer directly. If you are given Options, your answer should be one of them. Otherwise, your answer should be very brief. 

# Question: 
# {question}

# Answer: 
# {reasoning}
# """


# Search_PROMPT = """You are an advanced question-answering agent equipped with a specialized module to aid in analyzing and responding to queries about images:
# Search: This module performs web searches on the internet to gather relevant information for any query. It's especially useful for retrieving knowledge, definitions and problem-solving strategies. When you need this module, state your request as: "<|begin_search|> your query or topic of interest <|end_search|>".


# When faced with a question about an image, your task is to:
# 1. Reason Step-by-Step: Break down the problem into manageable steps, explaining your reasoning at each stage.
# 2. Utilize Module When Necessary: Assign specific tasks to each module as needed, based on their capabilities, to gather additional information essential for answering the question accurately. If web searches are required, state your request as: "<|begin_search|> your query or topic of interest <|end_search|>". 
# 3. Provide the Final Answer: After completing your reasoning and any necessary computations, provide your final answer in the format \\boxed{{YOUR_ANSWER}}.


# Here are some examples:
# Question1: What is the perimeter of the rectangle?

# Answer:
# From the image, we can see that the rectangle has dimensions: Length (l) = 5 ft, Width (w) = 3 ft. To find the perimeter of a rectangle, we need to search for the formula. <|begin_search|> formula of perimeter of a rectangle with given length and width. <|end_search|> <|begin_result|>The formula for the perimeter of a rectangle with a given length (L) and width (W) is: Perimeter = 2 \\times (L + W).<|end_result|> Therefore, the perimeter can be calculated as: 2 \\times (5 + 3) = 16 ft. The final answer is \\boxed{{16}}. 

# Question2: Find $x$ so that each quadrilateral is a parallelogram.

# Answer:
# The image shows a quadrilateral that resembles a parallelogram. The left side is 2x-5, and the right side is 3x-18. To determine the value of x that makes the quadrilateral a parallelogram, we need to use the property of a parallelogram to create an equation. <|begin_search|> Properties of parallelograms <|end_search|> <|begin_result|> In a parallelogram, opposite sides are equal. <|end_result|> Therefore, 2x-5=3x-18. So, x=13. The final answer is \\boxed{{13}}.


# Please refer to the prompts and examples above to help me solve the following problem. You should think step by step to solve it. Please strictly adhere to the choice format in the question and provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
# Question: {input}

# Answer: 
# {previous_reasoning}"""


# Search_PROMPT = """You are an advanced question-answering agent equipped with a specialized module to aid in analyzing and responding to queries about images:
# Search: This module performs web searches on the internet to gather relevant information for any query. It's especially useful for retrieving knowledge, definitions and problem-solving strategies. When you need this module, state your request as: "<|begin_search|> your query or topic of interest <|end_search|>".


# When faced with a question about an image, your task is to:
# 1. Reason step by step.
# 2. Utilize search module during your reasoning. When you need this module, use <|begin_search|> to request a web search and end with <|end_search|>.
# 3. Provide the final answer in the format \\boxed{{YOUR_ANSWER}}. If you are given Choices, your answer should be one of them. Otherwise, your answer should be plain numbers without any units.


# Here are some examples:
# Question1: 
# What is the perimeter of the rectangle?
# Choices:
# (A) 8
# (B) 12
# (C) 16
# (D) 20

# Answer:
# From the image, we can see that the rectangle has dimensions: Length (l) = 5 ft, Width (w) = 3 ft. To find the perimeter of a rectangle, we need to search for the formula. <|begin_search|> formula of perimeter of a rectangle with given length and width. <|end_search|> <|begin_result|>The formula for the perimeter of a rectangle with a given length (L) and width (W) is: Perimeter = 2 \\times (L + W).<|end_result|> Therefore, the perimeter can be calculated as: 2 \\times (5 + 3) = 16 ft. The final answer is \\boxed{{C}}. 


# Question2: 
# Find $x$ so that each quadrilateral is a parallelogram.

# Answer:
# The image shows a quadrilateral that resembles a parallelogram. The left side is 2x-5, and the right side is 3x-18. To determine the value of x that makes the quadrilateral a parallelogram, we need to use the property of a parallelogram to create an equation. <|begin_search|> Properties of parallelograms <|end_search|> <|begin_result|> In a parallelogram, opposite sides are equal. <|end_result|> Therefore, 2x-5=3x-18. So, x=13. The final answer is \\boxed{{13}}.


# Please refer to the prompts and examples above to help me solve the following problem. 
# Question: 
# {question}

# Answer: 
# {previous_reasoning}"""


# Code_PROMPT = """You are an advanced question-answering agent equipped with a specialized module to aid in analyzing and responding to questions about images:
# Code: This module allows you to write and execute Python code to perform tasks such as calculations, data analysis, simulations, or solving algorithmic problems. It's ideal when a task requires logic implementation and mathematical computation. When you need this module, state your Python code as: "<|begin_code|> your Python code <|end_code|>". Ensure your code is complete, syntactically correct, and uses proper Python naming conventions.


# When faced with a question about an image, your task is to:
# 1. Reason step by step.
# 2. Utilize code module during your reasoning. When you need this module, use <|begin_code|> to run Python code and end with <|end_code|>.
# 3. Provide the final answer in the format \\boxed{{YOUR_ANSWER}}. If you are given Choices, your answer should be one of them. Otherwise, your answer should be plain numbers without any units.


# Here is an example:
# Question:
# What is the sum of the first 50 positive integers?

# Answer: 
# To find the sum of the first 50 positive integers, we can use the formula for the sum of an arithmetic series:
# <|begin_code|>
# n = 50
# sum = n * (n + 1) / 2
# print(sum)
# <|end_code|>
# <|begin_result|>1275<|end_result|> Therefore, The final answer is \\boxed{{1275}}.


# Please refer to the prompts and examples above to help me solve the following problem. 
# Question: 
# {question}

# Answer: 
# {previous_reasoning}"""


# Perceive_PROMPT = """You are an advanced question-answering agent equipped with a specialized module to aid in analyzing and responding to questions about images:
# Perceive: This module can percerive visual content to answer simple questions about an image. It is especially useful for answering visual sub-questions such as identifying objects, counting instances, estimating attributes (e.g., age, color, size), and describing spatial relationships. Use this module when reasoning requires detailed understanding of the image. When you need this module, state your request as: "<|begin_perceive|> your question <|end_perceive|>".


# When faced with a question about an image, your task is to:
# 1. Reason step by step.
# 2. Utilize Perceive module during your reasoning. When you need this module, use "<|begin_perceive|> your question <|end_perceive|>".
# 3. Provide the final answer in the format \\boxed{{YOUR_ANSWER}}. If you are given Choices, your answer should be one of them. Otherwise, your answer should be plain numbers without any units.


# Here is an example:
# Question:
# What is the age gap between these two people in image?

# Answer: 
# To find the age gap between these two people, we need to find out the age for each of them.
# <|begin_perceive|> What is the approximate age of the person on the left? <|end_perceive|>
# <|begin_result|> Around 10 years old. <|end_result|> 
# Also, the person on the right is around 30 years old. Therefore, the final answer is \\boxed{{20}}.


# Please refer to the prompts and examples above to help me solve the following problem. 
# Question: 
# {question}

# Answer: 
# {previous_reasoning}"""


# Full_PROMPT = """You are an advanced question-answering agent equipped with some specialized modules to aid in analyzing and responding to questions about images:
# 1. Search: This module performs web searches on the internet to gather relevant information for any query. It's especially useful for retrieving knowledge, definitions and problem-solving strategies. When you need this module, state your request as: "<|begin_search|> your query or topic of interest <|end_search|>".
# 2. Code: This module allows you to write and execute Python code to perform tasks such as calculations, data analysis, simulations, or solving algorithmic problems. It's ideal when a task requires logic implementation and mathematical computation. When you need this module, state your Python code as: "<|begin_code|> your Python code <|end_code|>". Ensure your code is complete, syntactically correct, and uses proper Python naming conventions.
# 3. Perceive: This module can percerive visual content to answer simple questions about an image. It is especially useful for answering visual sub-questions such as identifying objects, counting instances, estimating attributes (e.g., age, color, size), and describing spatial relationships. Use this module when reasoning requires detailed understanding of the image. When you need this module, state your request as: "<|begin_perceive|> your question <|end_perceive|>".


# When faced with a question about an image, your task is to:
# 1. Reason step by step.
# 2. Utilize modules during your reasoning. When you need Search module, use <|begin_search|> to request a web search and end with <|end_search|>. When you need Code module, use <|begin_code|> to run Python code and end with <|end_code|>. When you need Perceive module, use <|begin_perceive|> to ask your question and end with <|end_perceive|>.
# 3. Provide the final answer in the format \\boxed{{YOUR_ANSWER}}. If you are given Choices, your answer should be one of them. Otherwise, your answer should be plain numbers without any units.


# Here are some examples:
# Question1: 
# What is the perimeter of the rectangle?
# Choices:
# (A) 8
# (B) 12
# (C) 16
# (D) 20

# Answer:
# From the image, we can see that the rectangle has dimensions: Length (l) = 5 ft, Width (w) = 3 ft. To find the perimeter of a rectangle, we need to search for the formula. <|begin_search|> formula of perimeter of a rectangle with given length and width. <|end_search|> <|begin_result|>The formula for the perimeter of a rectangle with a given length (L) and width (W) is: Perimeter = 2 \\times (L + W).<|end_result|> Therefore, the perimeter can be calculated as: 2 \\times (5 + 3) = 16 ft. The final answer is \\boxed{{C}}. 


# Question2:
# What is the sum of the first 50 positive integers?

# Answer: 
# To find the sum of the first 50 positive integers, we can use the formula for the sum of an arithmetic series:
# <|begin_code|>
# n = 50
# sum = n * (n + 1) / 2
# print(sum)
# <|end_code|>
# <|begin_result|>1275<|end_result|> Therefore, The final answer is \\boxed{{1275}}.


# Question3:
# What is the age gap between these two people in image?

# Answer: 
# To find the age gap between these two people, we need to find out the age for each of them.
# <|begin_perceive|> What is the approximate age of the person on the left? <|end_perceive|>
# <|begin_result|> Around 10 years old. <|end_result|> 
# Also, the person on the right is around 30 years old. Therefore, the final answer is \\boxed{{20}}.


# Please refer to the prompts and examples above to help me solve the following problem. 
# Question: 
# {question}

# Answer: 
# {previous_reasoning}"""


# 3. Integrate and rephrase the extracted information smoothly into the reasoning chain. Do not quote or copy verbatim. Instead, continue the assistant's reasoning using fluent, natural language. Phrase the information as internal knowledge or informed commentary, using expressions like "According to external sources", "As is known", or "It is generally established that...".

# Integrate_PROMPT = """You are an advanced reasoning agent. Given the Question about an image, the Previous Reasoning Steps, a Current Search Query, and a set of Searched Documents, your task is to:
# 1. Analyze the Previous Reasoning Steps and Current Search Query to understand the assistant's current objective and what specific information is required.
# 2. Read Searched Documents and identify content directly relevant to the Current Search Query.
# 3. Integrate and rephrase the extracted information smoothly into the reasoning chain. Do not quote or copy verbatim. Instead, answer the Current Search Query using fluent, natural language. Phrase the information as internal knowledge or informed commentary, using expressions like "According to external sources", "As is known", or "As the search results show".


# Here is an example:
# Question:
# Find $x$ so that each quadrilateral is a parallelogram.

# Previous Reasoning Steps:
# The image shows a quadrilateral that resembles a parallelogram. The left side is 2x-5, and the right side is 3x-18. To determine the value of x that makes the quadrilateral a parallelogram, we need to use the property of a parallelogram to create an equation.

# Current Search Query:
# Properties of parallelograms

# Searched Documents:
# Passage 1: 
# A simple (non-self-intersecting) quadrilateral is a parallelogram if and only if any one of the following statements is true:
# Two pairs of opposite sides are parallel (by definition).
# Two pairs of opposite sides are equal in length.
# Two pairs of opposite angles are equal in measure.
# The diagonals bisect each other.
# One pair of opposite sides is parallel and equal in length.
# Adjacent angles are supplementary.

# Output:
# In a parallelogram, opposite sides are equal.


# Now complete the task for the input below:
# Question:
# {question}

# Previous Reasoning Steps:
# {previous_reasoning}

# Current Search Query:
# {calling}

# Searched Documents:
# {raw_result}

# Output:
# """


# Perceive_Module_PROMPT = """You are an advanced visual expert. Given an image and a question, your task is to perceive the visual content carefully and provide an accurate, concise answer. Focus on what can be directly observed or reasonably inferred from the image.

# Question: 
# {question}
# """



# PROMPT = """You are a reasoning assistant with the ability to use tools to help you answer the user's question accurately. The results of a certain tool will be given in the format <|begin_result|> ...tool results... <|end_result|>.

# TOOLS:
# ------

# You have access to the following tools:
# > Web search: useful for when you need to perform a web search to get external knowledge. To perform a web search: write <|begin_search|> your query here <|end_search|>.
# > Image captioning: useful when you want to know what is inside the photo. To perform image captioning, write <|begin_captioning|> your query here <|end_captioning|>.
# > Image object detection: useful when you want to know what objects are in the photo. To perform image object detection, write <|begin_detection|> <|end_detection|>.

# ------

# Examples:
# Question: What is the perimeter of the rectangle?
# Assitant: To find the perimeter of a rectangle, let's calculate the perimeter using the formula.\n<|begin_search|>calculate perimeter of a rectangle with given length and width<|end_search|>: 

# ------

# Begin!

# Question: {input}
# {previous_reasoning}"""

# PROMPT = """You are a reasoning assistant with the ability to use tools to help you answer the user's question accurately. The results of a certain tool will be given in the format <|begin_result|> ...tool results... <|end_result|>.

# TOOLS:
# ------

# You have access to the following tools:
# > Web search: useful for when you need to perform a web search to get external knowledge. To perform a web search: write <|begin_search|> your query here <|end_search|>.

# ------

# Examples:
# Question: What is the perimeter of the rectangle?
# Assitant: To find the perimeter of a rectangle, let's calculate the perimeter using the formula.\n<|begin_search|>calculate perimeter of a rectangle with given length and width<|end_search|>: 

# ------

# Begin!

# Question: {input}
# {previous_reasoning}"""


# def get_math_prompt(question, previous_reasoning):
#     """
#     Generates a prompt for the math reasoning task.
    
#     Args:
#         question (str): The math question to be answered.
#         previous_reasoning (str): Any previous reasoning or context to be included in the prompt.

#     Returns:
#         str: The formatted prompt for the math reasoning task.
#     """
#     task_instruction = (
#         'Please answer the following math question. You should think step by step to solve it. Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n'
#         f'Question: {question}'
#     )
#     user_prompt = PROMPT.format(input=task_instruction, previous_reasoning=previous_reasoning)
#     return user_prompt


# def get_refine_prompt(previous_reasoning: str, search_query: str, document: str) -> str:
#     return f"""**Task Instruction:**

# You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

# **Guidelines:**

# 1. **Analyze the Searched Web Pages:**
# - Carefully review the content of each searched web page.
# - Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

# 2. **Extract Relevant Information:**
# - Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
# - Ensure that the extracted information is accurate and relevant.

# 3. **Output Format:**
# - **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
# **Final Information**

# [Helpful information]

# - **If the web pages do not provide any helpful information for current search query:** Output the following text.

# **Final Information**

# No helpful information found.

# **Inputs:**
# - **Previous Reasoning Steps:**  
# {previous_reasoning}

# - **Current Search Query:**  
# {search_query}

# - **Searched Web Pages:**  
# {document}

# Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
# """


# def get_math_search_o1_instruction():
#     return (
#         "You are a reasoning assistant with the ability to perform web searches to help "
#         "you answer the user's question accurately. You have special tools:\n\n"
#         "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
#         "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
#         f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to 10.\n\n"
#         "Once you have all the information you need, continue your reasoning.\n\n"
#         "Example:\n"
#         "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
#         "Assistant thinking steps:\n"
#         "- I might need to look up techniques for integrating e^(x^2).\n\n"
#         "Assistant:\n"
#         "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
#         "(System returns processed information from relevant web pages)\n\n"
#         "Assistant continues reasoning with the new information...\n\n"
#         "Remember:\n"
#         "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
#         "- When done searching, continue your reasoning.\n\n"
#     )


# def get_task_instruction_math(question, model_name=None):
#     user_prompt = (
#         'Please answer the following math question. You should think step by step to solve it.\n\n'
#         'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
#         f'Question:\n{question}\n\n'
#     )
#     return user_prompt


# def get_naive_rag_instruction(question, documents):
#     return (
#         "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
#         "Question:\n"
#         f"{question}\n"
#         "Documents:\n"
#         f"{documents}\n"
#     )

