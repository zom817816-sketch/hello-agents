import re 
from typing import Optional, List, Tuple
from .my_agent import MyAgent
from .my_llm import MyLLM
from .my_config import Config
from .my_messages import Message
from .my_tools import ToolRegistry

MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""

class ReActAgent(MyAgent): 
    """
    ReAct Agent - 推理与行动结合的智能体
    """

    def __init__(
        self, 
        name: str, 
        llm: MyLLM, 
        tool_registry: ToolRegistry, 
        system_prompt: Optional[str] = None, 
        config: Optional[Config] = None, 
        max_steps: int = 5, 
        custom_prompt: Optional[str] = None
    ): 
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")
    
    def run(self, input_text: str, **kwargs) -> str: 
        """
        运行ReAct Agent 
        """
        self.current_history = []
        current_step = 0 

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps: 
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 构建提示词
            tools_description = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_description, 
                question=input_text, 
                history=history_str
            )

            # 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.think(messages, **kwargs)

            # 解析输出
            thought, action = self._parse_output(response_text)

            # 检查完成条件
            if action and action.startswith("Finish"): 
                final_answer = self._parse_action(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer
            
            # 执行工具调用
            if action: 
                tool_name, tool_input = self._parse_action(action)
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")
        
        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。" 
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
        

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

    