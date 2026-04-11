import os 
from typing import Optional
from openai import OpenAI
from llm_client import MyAgentsLLM

class MyLLM(MyAgentsLLM): 
    """ 
    自定义的LLM客户端，支持多种模型提供商和自动差异化配置
    """
    # 提供商默认配置
    PROVIDER_CONFIGS = {
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-3.5-turbo"
        },
        "doubao": {
            "api_key_env": "DOUBAO_API_KEY",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "default_model": "doubao-seed-2-0-mini-260215"
        },
        "modelscope": {
            "api_key_env": "MODELSCOPE_API_KEY",
            "base_url": "https://api-inference.modelscope.cn/v1/",
            "default_model": "Qwen/Qwen2.5-VL-72B-Instruct"
        },
        "zhipu": {
            "api_key_env": "ZHIPU_API_KEY",
            "base_url": "https://open.bigmodel.cn/api/mock/gpt",
            "default_model": "glm-4"
        },
        "ollama": {
            "api_key_env": "OLLAMA_API_KEY",
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama3"
        },
        "vllm": {
            "api_key_env": "VLLM_API_KEY",
            "base_url": "http://localhost:8000/v1",
            "default_model": "meta-llama/Llama-2-7b-chat-hf"
        },
        "deepseek": {
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat"
        },
        "anthropic": {
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com/v1",
            "default_model": "claude-3-opus-20240229"
        },
        "google": {
            "api_key_env": "GOOGLE_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1",
            "default_model": "gemini-pro"
        }
    }

    def __init__(
        self, 
        model: Optional[str] = None, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        provider: Optional[str] = "auto", 
        **kwargs
    ): 
        # 自动检测提供商
        if provider == "auto":
            provider = self._auto_detect_provider(api_key, base_url)
            print(f"自动检测到提供商: {provider}")

        # 解析凭证和配置
        self.provider = provider
        resolved_api_key, resolved_base_url, resolved_model = self._resolve_config(model, api_key, base_url)

        # 验证凭证是否存在
        if not resolved_api_key:
            raise ValueError(f"{provider} API key not found. Please set {self.PROVIDER_CONFIGS.get(provider, {}).get('api_key_env', 'API_KEY')} environment variable.")

        # 设置参数
        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self.model = resolved_model
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens")
        self.timeout = kwargs.get("timeout", 60)

        # 使用获取的参数创建OpenAI客户端实例
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url, 
            timeout=self.timeout
        )

        print(f"成功初始化 {provider} 提供商，使用模型: {self.model}")
    
    def _auto_detect_provider(
        self, 
        api_key: Optional[str], 
        base_url: Optional[str]
    ) -> str: 
        """ 
        自动检测LLM提供商
        """ 
        # 1.检查特定提供商的环境变量
        for provider, config in self.PROVIDER_CONFIGS.items():
            if os.getenv(config["api_key_env"]):
                return provider
        
        # 检查通用环境变量
        if os.getenv("LLM_API_KEY"):
            return "openai" # 默认使用openai

        # 2. 根据base_url判断
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url: 
            base_url_lower = actual_base_url.lower() 
            if "api-inference.modelscope.cn" in base_url_lower: return "modelscope"
            if "ark.cn-beijing.volces.com" in base_url_lower: return "doubao"
            if "open.bigmodel.cn" in base_url_lower: return "zhipu" 
            if "deepseek.com" in base_url_lower: return "deepseek"
            if "anthropic.com" in base_url_lower: return "anthropic"
            if "generativelanguage.googleapis.com" in base_url_lower: return "google"
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower: 
                if ":11434" in base_url_lower: return "ollama" 
                if ":8000" in base_url_lower: return "vllm" 
                return "local" # 其他本地端口 
        
        # 3. 根据API 密钥格式判断
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        if actual_api_key: 
            if actual_api_key.startswith("ms-"): return "modelscope"
            if actual_api_key.startswith("sk-"): return "openai"
            if actual_api_key.startswith("pk-"): return "zhipu"
            if actual_api_key.startswith("ds-"): return "deepseek"
            if actual_api_key.startswith("anthropic-"): return "anthropic"
        
        # 4. 默认返回 'openai'，使用通用配置
        return "openai"
    
    def _resolve_config(self, model: Optional[str], api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str, str]:
        """根据provider解析API密钥、base_url和模型"""
        # 获取提供商配置
        config = self.PROVIDER_CONFIGS.get(self.provider, {})
        
        # 解析API密钥
        resolved_api_key = api_key or os.getenv(config.get("api_key_env")) or os.getenv("LLM_API_KEY")
        
        # 解析base_url
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or config.get("base_url")
        
        # 解析模型
        resolved_model = model or os.getenv("LLM_MODEL_ID") or config.get("default_model")
        
        return resolved_api_key, resolved_base_url, resolved_model

    def switch_provider(self, provider: str, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """切换到其他提供商"""
        # 重新初始化
        self.__init__(
            model=model or self.model,
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            provider=provider,
            **kwargs
        )
        return f"已切换到 {provider} 提供商，使用模型: {self.model}"
