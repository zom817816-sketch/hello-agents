import os 
from typing import Optional
from openai import OpenAI
from llm_client import MyAgentsLLM

class MyLLM(MyAgentsLLM): 
    """ 
    自定义的LLM客户端，通过继承增加对doubao的支持
    """
    def __init__(
        self, 
        model: Optional[str] = None, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        provider: Optional[str] = "auto", 
        **kwargs
    ): 
        # 检查provider是否为我们想处理的“modelscope”
        if provider == "doubao": 
            print("正在使用自定义的doubao Provider")
            self.provider = "doubao"

            # 解析ModelScope的凭证
            self.api_key = api_key or os.getenv("DOUBAO_API_KEY")
            self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3"

            # 验证凭证是否存在
            if not self.api_key: 
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "doubao-seed-2-0-mini-260215"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            # 使用获取的参数创建OpenAI客户端实例
            # 使用self.client以保持与父类一致
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url=self.base_url, 
                timeout=self.timeout
            )

        else: 
            # 如果不是modelscope，则按照父类的原始逻辑处理
            # 从kwargs中提取timeout参数
            timeout = kwargs.get('timeout', 60)
            super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout)
        

