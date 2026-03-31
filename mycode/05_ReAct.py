""" 
ReAct通过特殊的提示工程引导模型，使其每一步输出都遵循一个固定的轨迹: 
- Thought 
- Act 
- Observation 
""" 

from zai import ZhipuAiClient
import os
from openai import OpenAI 
import re
from dotenv import load_dotenv 
from typing import List, Dict, Any 
from LLMClient import MyAgentsLLM

# 加载.env文件中的环境变量 
load_dotenv()

# 实现搜索工具 
# 良好定义的工具应具有三要素：名称、描述、执行逻辑 
def search(query: str) -> str: 
    """ 
    基于ZhipuAI的web search工具 
    解析搜索结果并返回直接答案 
    """ 
    print(f'正在执行网络搜索: {query}') 
    try: 
        api_key = os.getenv("ZAI_API_KEY") 
        client = ZhipuAiClient(api_key=api_key) 
        response = client.web_search.web_search(
            search_engine="search_std", 
            search_query=query, 
            count=3,
            search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
            content_size="medium"  # 控制网页摘要的字数，默认medium
        ) 
        output = ""
        for i, result in enumerate(response.search_result): 
            output += (f"[{i+1}] {result.title}\n{result.content}\n\n")  
        return output if output else f"未找到'{query}'的信息"
    except Exception as e: 
        return f"网络搜索时发生错误: {e}"
    
class ToolExecuter: 
    """
    一个工具执行器，负责管理和执行工具
    """ 
    def __init__(self): 
        self.tools: Dict[str, Dict[str, Any]] = {} 
    
    def registerTool(self, func: callable, name: str, description: str): 
        """ 
        向工具列表中注册一个工具
        """ 
        if name in self.tools: 
            print(f"{name}工具存在，将被覆盖") 
        self.tools[name] = {'func': func, 'description': description} 
        print(f"{name}工具已注册") 
    
    def getTool(self, name: str): 
        """ 
        根据name查询对应的工具的执行函数 
        """ 
        return self.tools.get(name, {}).get('func') 
    
    def getAvailableTools(self): 
        """ 
        获取所有可用工具的格式化描述字符串
        """ 
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )

# ReAct 智能体实现 
# 定义提示词模板
REACT_PROMPT_TEMPLATE = """
你是一个有能力调用外部工具的智能助手。
可用工具如下:
{tools}

请严格按照下列格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- tool_name[tool_input]: 调用一个可用工具，不要加反引号
- Finish[最终答案]: 当你认为已获得最终答案时，不要加反引号

示例：
Thought: 我需要搜索最新信息
Action: Search[OpenAI最新模型]

现在请解决以下问题：
Question: {question}
History: {history}
"""

# ReActAgent实现 
class ReActAgent: 
    def __init__(self, llm_client: MyAgentsLLM, tool_executer: ToolExecuter, max_steps: int=5): 
        self.llm_client = llm_client 
        self.tool_executer = tool_executer 
        self.max_steps = max_steps 
        self.history = [] 
    
    def run(self, question: str): 
        """ 
        运行ReAct智能体回答一个问题 
        """ 
        self.history = [] # 每次运行时重置历史记录 

        for step in range(self.max_steps): 
            print(f'---- 第{step+1}步 ----') 
            
            # 格式化提示词 
            tools_desc = self.tool_executer.getAvailableTools() 
            history_str = "\n".join(self.history) 
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools = tools_desc, 
                question = question, 
                history = history_str
            )

            # 调用LLM进行思考 
            messages = [{'role': 'user', 'content': prompt}] 
            response = self.llm_client.think(messages) 
            if not response: 
                print("错误: LLM未能返回有效响应") 
                break 

            # 解析LLM的输出 
            thought, action = self._parse_output(response) 
            if thought: 
                print(f"🤔思考: {thought}") 
            if not action: 
                print("警告: 未能解析出有效的Action，流程终止。") 
                break 

            # 执行Action 
            if action.startswith('Finish'): 
                # 提取最终答案 
                match = re.match(r"Finish\[(.*)\]", action)
                if match:
                    final_answer = match.group(1) 
                    print(f"最终答案: {final_answer}") 
                    return final_answer
                else:
                    print(f"警告: Finish格式不正确: {action}")
                    break 
            
            tool_name, tool_input = self._parse_action(action) 
            if not tool_name or not tool_input: 
                # 处理无效的Action格式 
                continue 

            print(f"🎬行动: {tool_name}[{tool_input}]")

            tool_func = self.tool_executer.getTool(tool_name) 
            if not tool_func: 
                observation = f"未找到名为{tool_name}的工具" 
            else: 
                observation = tool_func(tool_input) # 调用真实工具 
            print(f"👀观察: {observation}") 

            # 将本轮的Action和Observation放到历史中 
            self.history.append(f"Action: {action}") 
            self.history.append(f"Observation: {observation}") 
        
        # 循环结束 
        print("已达到最大步数，流程终止。") 
        return None 

    
    def _parse_output(self, text: str): 
        """
        解析LLM的输出，提取Thought和Action 
        """ 
        # Thought:匹配到Action:或文本末尾 
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)",
            text, 
            re.DOTALL
        ) 
        # Action:匹配到文本末尾 
        action_match = re.search(
            r"Action:\s*(.*?)$", 
            text, 
            re.DOTALL
        ) 
        thought = thought_match.group(1).strip() if thought_match else None 
        action = action_match.group(1).strip() if action_match else None 
        return thought, action 
    
    def _parse_action(self, action_text: str): 
        """ 
        解析Action字符串，获取工具名称和工具输入 
        """ 
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL) 
        return (match.group(1), match.group(2)) if match else (None, None) 

# ReAct的特点:高可解释性，动态规划与纠错能力，工具协同能力 
# 局限性:对LLM自身能力的强依赖能力，执行效率问题，提示词的脆弱性，可能陷入局部最优 
# 调试技巧:检查完整的提示词，分析原始输出，验证工具的输入与输出，调整提示词中的示例，尝试不同的参数和模型


if __name__ == "__main__": 
    # toolexecuter = ToolExecuter()
    # tools_desc = "一个网络搜索引擎，当你需要回答事实、时事以及在你的知识库中找不到的信息时，请使用此工具" 
    # toolexecuter.registerTool(func=search, name="search", description=tools_desc) 
    # # 获取可用的工具 
    # print("\n -- 可用工具 --") 
    # print(toolexecuter.getAvailableTools()) 
    # # 测试工具 
    # name = 'search'
    # func = toolexecuter.getTool(name=name) 
    # query = "英伟达的最强游戏显卡是什么"
    # print(f"\n执行搜索工具搜索: {query}")
    # if func: 
    #     output = func(query)
    #     print(output) 
    # else: 
    #     print(f"未找到名为: {name}的工具") 

    # ReAct智能体测试 
    tool_executer = ToolExecuter() 
    tool_executer.registerTool(
        func=search, 
        name='Search', 
        description="一个网络搜索引擎，当你需要回答事实、时事以及在你的知识库中找不到的信息时，请使用此工具"
    ) 
    client = MyAgentsLLM(timeout=30)
    agent = ReActAgent(client, tool_executer) 
    query = "OpenAI目前最新发布的大模型是什么" 
    final_answer = agent.run(question=query) 
