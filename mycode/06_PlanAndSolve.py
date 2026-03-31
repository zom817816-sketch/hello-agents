import os
from openai import OpenAI 
import re
from dotenv import load_dotenv 
from typing import List, Dict, Any 
import ast
from LLMClient import MyAgentsLLM

# 加载.env文件中的环境变量 
load_dotenv()

# Plan and Solve 
# 两个阶段:先规划，后行动 

PLANNER_PROMPT_TEMPLATE = """ 
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
""" 

# 这个提示词确保了输出的质量和稳定性:角色描述，任务描述，格式约束 

# Planner 
class Planner: 
    def __init__(self, llm_client: MyAgentsLLM): 
        self.client = llm_client 

    def plan(self, question: str) -> list: 
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question) 
        messages = [
            {'role':'user', 'content': prompt}
        ]
        response = self.client.think(messages) or "" 
        print(f"✅ 计划已生成:\n{response}") 

        # 解析LLM输出的字符串 
        try: 
            # 找到```python ```之间的内容 
            plan_str = response.split("```python")[1].split("```")[0].strip() 
            # 使用ast.literal_eval()安全执行字符串 
            plan = ast.literal_eval(plan_str) 
            return plan if type(plan) == list else [] 
        except (ValueError, SyntaxError, IndexError) as e: 
            print(f"解析计划时出错: {e}") 
            print(f"原始响应: {response}") 
            return [] 
        except Exception as e: 
            print(f"解析计划时出现未知错误: {e}") 
            return [] 

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
""" 

# Executor 
class Executor: 
    def __init__(self, llm_client: MyAgentsLLM): 
        self.client = llm_client 
    def execute(self, question: str, plan: list[str]) -> str: 
        """
        根据计划逐步执行
        """
        history = "" # 存储执行历史 
        for i, action in enumerate(plan): 
            current_step = f"第{i+1}步: {action}"
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, 
                plan = plan, 
                history = history, 
                current_step = current_step
            ) 
            messages = [{'role': 'user', 'content': prompt}]
            response = self.client.think(messages) or ""
            # 更新历史记录
            history += f" 第{i+1}步:{action}\n结果: {response}" 

            print(f"✅ 步骤 {i+1} 已完成，结果: {response}") 
        
        # 循环结束后返回最后响应(答案) 
        return response 

# PlanAndSolveAgent 
class PlanAndSolveAgent: 
    def __init__(self, llm_client: MyAgentsLLM): 
        self.planner = Planner(llm_client) 
        self.executor = Executor(llm_client) 
    
    def run(self, question): 
        """ 
        先规划，后执行 
        """ 
        print("---- 正在进行任务规划 ----")
        plan = self.planner.plan(question) 
        if not plan: 
            print(f"\n未生成有效规划")
        print("---- 正在按规划执行 ----")
        answer = self.executor.execute(question, plan) 
        print(f"\n任务完成，最终答案为: {answer}") 
        return answer 

if __name__ == "__main__": 
    llm_client = MyAgentsLLM(timeout=30) 
    plan_and_solve_agent = PlanAndSolveAgent(llm_client) 
    question = "问题: 一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    answer = plan_and_solve_agent.run(question)


