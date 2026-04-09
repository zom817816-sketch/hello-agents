import os
from openai import OpenAI 
from dotenv import load_dotenv 
from typing import List, Dict, Any, Optional

# 加载.env文件中的环境变量 
load_dotenv()

class MyAgentsLLM: 
    """ 
    兼容OpenAI接口的服务，默认使用流式响应
    """ 
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: Optional[int] = None): 
        self.provider = provider or os.getenv('PROVIDER')
        self.model = model or os.getenv('LLM_MODEL_ID')
        api_key = api_key or os.getenv('API_KEY') 
        base_url = base_url or os.getenv('BASE_URL') 
        timeout = timeout or 60 
        if not all([self.model, api_key, base_url, timeout]): 
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url, 
            timeout=timeout
        ) 

    def think(self, messages: List[Dict[str, str]], temperature: float=0.0) -> str: 
        """ 
        调用LLM进行思考,并返回响应
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages, 
                model=self.model, 
                temperature=temperature,
                stream=True
            ) 

            # 处理流式响应
            collected_content = [] 
            for chunk in response: 
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True) 
                collected_content.append(content) 
            print() # 流式输出后换行
            return collected_content

        except Exception as e: 
            print(f'调用LLM时发生错误: {e}') 
            return None 