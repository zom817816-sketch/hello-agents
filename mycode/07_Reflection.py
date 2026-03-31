"""为智能体引入一种事后（post-hoc）的自我校正循环，使其能够像人类一样，审视自己的工作，发现不足，并进行迭代优化
核心工作流程: 
执行: 首先使用ReAct 或 Plan-and-Solve尝试完成任务，生成一个初步的解决方案或者行动轨迹 
反思: 调用一个独立的、或者带有特殊提示词的大语言模型实例评估:事实性错误，逻辑漏洞，效率问题，遗漏问题
优化: 将初稿和反馈作为新的上下文，再次调用大模型根据反馈对初稿进行修正
""" 

from typing import List, Dict, Any, Optional 
from LLMClient import MyAgentsLLM

class Memory: 
    """
    短期记忆模块，用于存储智能体的行动与思考轨迹
    """ 
    def __init__(self): 
        self.records: List[Dict[str, Any]] = [] 

    def add_record(self, record_type, record_content): 
        """ 
        向记忆中添加一条记忆

        参数: 
        - record_type (str): 记录的具体类型('execution'|'reflection') 
        - content (str): 记录的具体内容(生成的代码或者反馈)
        """ 
        record = {'type': record_type, 'content': record_content} 
        self.records.append(record) 
        print(f"已添加类型为{record_type}的一条记录到记忆中") 
    
    def get_trajectory(self) -> str: 
        """ 
        将所有记忆转换为一个连贯的字符串文本用于构建提示词 
        """ 
        trajectory_parts = [] 
        for record in self.records: 
            if record['type'] == 'execution': 
                trajectory_parts.append(f"--- 上一轮尝试(代码) ---\n{record['content']}") 
            elif record['type'] == 'reflection': 
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}") 
        
        return "\n\n".join(trajectory_parts) 
    
    def get_last_execution(self) -> Optional[str]: 
        """ 
        返回最近一次的执行结果，如果没有则返回None 
        """ 
        for record in self.records: 
            if record['type'] == 'execution': 
                return record['content'] 
        return None
    
# 提示词 
# 初始执行提示词
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

# 反思提示词 
REFLECT_PROMPT_TEMPLATE = """ 
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
""" 

# 优化提示词
REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

# 智能体封装与实现 
class ReflectionAgent: 
    """ 
    一个引入反思机制持续优化的智能体
    """ 
    def __init__(self, llm_client: MyAgentsLLM, memory: Memory, max_steps: int=3): 
        self.llm_client = llm_client 
        self.memory = memory
        self.max_steps = max_steps 

    def run(self, task): 
        print(f'\n--- 正在处理任务: {task} ---')
        # 先获取初始响应
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task) 
        initial_ouput = self._get_response(initial_prompt) 
        self.memory.add_record('execution', initial_ouput)
        print(f"\n--- 初始响应: {initial_ouput} ---") 

        # 进行反思迭代优化 
        for step in range(self.max_steps): 
            print(f"\n--- 正在进行第{step+1}/{self.max_steps}轮优化 ---")

            # 反思 
            print("\n--- 正在进行反思 ---")
            last_code = self.memory.get_last_execution() 
            reflection_prompt = REFLECT_PROMPT_TEMPLATE.format(
                task=task, 
                code=last_code
            ) 
            feed_back = self._get_response(reflection_prompt) 
            self.memory.add_record('reflection', feed_back)

            # 检查是否需要停止
            if "无需改进" in feed_back: 
                print("\n 反思已认为当前代码无需改进,任务完成")
                break

            # 优化
            print("\n--- 正在进行优化 ---")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task, 
                last_code_attempt=last_code, 
                feedback=feed_back
            ) 
            refined_code = self._get_response(refine_prompt) 
            self.memory.add_record('execution', refined_code) 

        final_code = self.memory.get_last_execution()
        print(f"\n --- 任务完成 ---\n最终生成的代码:\n```Python\n{final_code}\n```")
        return final_code
 
    def _get_response(self, prompt: str) -> str:
        """ 
        用于获取LLM响应 
        """ 
        messages = [{'role': 'user', 'content': prompt}] 
        response = self.llm_client.think(messages) 
        return response if response else None

# Reflection成本:模型调用开销增加;任务延迟显著增加;提示工程复杂度上升
# Reflection收益:解决方案质量提升;鲁棒性与可靠性 


if __name__ == "__main__":
    llm_client = MyAgentsLLM(timeout=30) 
    memory = Memory() 
    reflection_agent = ReflectionAgent(llm_client,memory) 
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)"
    final_answer = reflection_agent.run(task)