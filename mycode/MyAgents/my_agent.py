"""Agent基类"""
from abc import ABC, abstractclassmethod
from typing import Optional, Any
from .my_messages import Message
from .my_llm import MyLLM
from .my_config import Config

class MyAgent(ABC): 
    """Agent基类"""

    def __init__(
        self, 
        name: str, 
        llm: MyLLM, 
        system_prompt: Optional[str] = None, 
        config: Optional[Config] = None
    ): 
        self.name = name 
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = [] # 存储对话历史记录
    
    @abstractclassmethod # @abstractclassmethod标记此方法为抽象方法，子类必须实现，否则无法实例化
    def run(seed, input_text: str, **kwargs) -> str: 
        """ 运行Agent """
        pass
    
    def add_message(self, message): 
        """ 添加消息到消息历史 """
        self._history.append(message)
    
    def clear_history(self): 
        """ 清空历史记录 """
        self._history.clear()
    
    def get_history(self) -> list[Message]: 
        """ 获取历史记录 """
        return self._history[:]
    
    def __str__(self) -> str: 
        return f"Agent(name={self.name}, provider={self.llm.provider})"

