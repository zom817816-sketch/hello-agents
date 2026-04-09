from dotenv import load_dotenv
from my_llm import MyLLM # 导入自己的类

# 加载环境变量
load_dotenv()

# 实例化重写后的客户端
llm = MyLLM(provider="modelscope")

# 准备消息
messages = [{"role": "user", "content": "你好，请介绍一下你自己"}]

# 发起调用，think方法都从父类继承，无需重写
response_stream = llm.think(messages=messages)

# 打印响应
print("DouBao Response:")
for chunk in response_stream: 
    # chunk在my_llm库中已经打印过一遍，这里只需pass
    # print(chunk, end="", flush=True)
    pass 

# 使用本地的VLLM
llm_vllm = MyLLM(
    provider="vllm", 
    model="Qwen/Qwen1.5-0.5B-Chat", # 需要与服务启动时指定的模型一致
    base_url="http://localhost:8000/v1", 
    api_key="vllm" # 本地服务一般不需要真实API KEY，可填任意非空字符串
)

# 使用本地的Ollama
llm_ollama = HelloAgentsLLM(
    provider="ollama",
    model="llama3", # 需与 `ollama run` 指定的模型一致
    base_url="http://localhost:11434/v1",
    api_key="ollama" # 本地服务同样不需要真实 Key
)

