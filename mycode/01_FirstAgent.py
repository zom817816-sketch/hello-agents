import requests 
import os
from tavily import TavilyClient
from dotenv import load_dotenv
from openai import OpenAI 
import re

load_dotenv()


AGENT_SYSTEM_PROMPT ="""
你是一个智能旅游助手,你的任务是分析用户请求，并一步步解决问题。

# 可用工具 
- `get_weather(sity: str)`:查询指定城市的天气
- `get_attraction(city: str, weather: str)`:根据城市和天气搜索推荐的旅游景点

# 输出格式
你每次回复必须为下列格式,包含一对Thought和Action

Thought: [你的思考过程和下一步]
Action: [你要执行的具体行动]

Action的格式必须是以下之一
1. 调用工具: function_name(args_name="args_value") 
2. 结束任务: Finish[最终答案] 

# 重要提示 
- 每次只输出一对Thought-Action 
- Action必须在同一行
- 收集到足够信息可以回答用户问题时,必须使用Action: Finish[最终答案] 格式结束

请开始吧! 
"""


# 查询天气
def get_weather(city: str) -> str: 
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """ 

    # API端点，请求JSON格式的数据 
    url = "https://wttr.in/{city}?format=j1" 
    
    try: 
        # 发起网络请求
        response = requests.get(url) 
        # 检查状态码 
        response.raise_for_status()
        # 解析返回的JSON数据 
        data = response.json() 

        # 提取当前的天气状况 
        current_condition = data['current_condition'][0] 
        weather_desc = current_condition['weather_desc'][0]['value'] 
        temp_c = current_condition['temp_C'] 

        # 格式化后返回
        return f"{city}当前天气为{weather_desc},气温为{temp_c}摄氏度"

    except requests.exceptions.RequestException as e: 
        # 处理网络错误 
        return f"查询天气时遇到网络问题 - {e}" 
    except (KeyError, IndexError) as e: 
        # 处理数据解析错误 
        return f"解析天气数据失败 - {e}"


# 根据城市和天气搜索推荐的旅游景点 
def get_attraction(city: str, weather: str) -> str: 
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """    

    api_key = os.getenv("TAVILY_API_KEY") 

    tavily = TavilyClient(api_key=api_key) 

    query = f"'{city}'在'{weather}'天气下适合去的旅游景点推荐及理由" 
    try:
        response = tavily.search(query, search_depth='basic', include_answer=True) 

        # Tavily返回的结果已经非常干净，可以直接使用
        if response.get("answer"): 
            return response["answer"] 
        
        # 如果没有综合回答，则格式化原始结果 
        formatted_results = [] 
        for result in response.get("results", []): 
            formatted_result = f"- {result['title']}: {result['content']}" 
            formatted_results.append(formatted_result) 
        
        if not formatted_results: 
            return f"未检索到相关的旅游景点推荐" 
        
        return "以下是为您检索到的景点信息:\n" + "\n".join(formatted_results) 

    except Exception as e: 
        return f"执行搜索时遇到错误: - {e}" 

available_tools = {
    "get_weather": get_weather, 
    "get_attraction": get_attraction
}

class OpenAICompatibleClient: 
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str): 
        self.model = model 
        self.client = OpenAI(api_key=api_key, base_url=base_url) 

    def generate(self, prompt: str, system_prompt: str) -> str: 
        """调用LLM API来产生回应""" 
        print("正在调用大语言模型") 
        try: 
            messages = [
                {'role': 'system', 'content': system_prompt}, 
                {'role': 'user', 'content': prompt}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages, 
                model=self.model, 
                stream=False 
            ) 

            print("大语言模型响应成功")
            return response.choices[0].message.content
        except Exception as e: 
            print(f'调用模型出现错误: - {e}') 
            return f'调用模型出错' 
         
# 请根据您使用的服务，将这里替换成对应的凭证和地址
API_KEY = os.getenv(API_KEY)
BASE_URL = os.getenv(BASE_URL)
MODEL_NAME = os.getenv(MODEL)
TAVILY_API_KEY= os.getenv(TAVILY_API_KEY) 

llm = OpenAICompatibleClient(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL) 

# 初始化 
user_prompt = "你好帮我查询一下北京天气，根据天气推荐适合的旅游景点" 
prompt_history = [f"用户请求: {user_prompt}"] 

# 运行主循环 
for i in range(5): 
    print(f'第{i+1}次循环') 
    full_prompt = "\n".join(prompt_history) 
    # 调用LLM
    ouput = llm.generate(prompt=full_prompt, system_prompt=AGENT_SYSTEM_PROMPT) 
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', ouput, re.DOTALL)
    if match: 
        truncated = match.group(1).strip() 
        if truncated != ouput: 
            ouput = truncated 
            print("已截断多余的Thought-Action对")
    print(f"模型输出: - {output}") 
    prompt_history.append(ouput) 

    # 解析并执行行动 
    action_match = re.search(r"Action: (.*)", ouput, re.DOTALL) 
    if not action_match: 
        observation_str = f"Observation: - 错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        print(f"{observation_str}\n" + "=" * 40) 
        prompt_history.append(observation_str) 
    action_str = action_match.group(1).strip() 

    if action_str.startswith("Finish"): 
        answer = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL).group(1)
        print(f"任务完成,最终答案为: - {answer}") 
        break 
    
    function_name = re.search(r"(\w+)\(", action_str).group(1) 
    function_args = re.search(r"\((.*)\)", action_str).group(1) 
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', function_args)) 

    if function_name not in available_tools: 
        observation = f"错误:未定义的工具: -{function_name}" 
    else:
        observation = available_tools[function_name](**kwargs) 
    
    # 记录观察结果 
    observation_str = f"Observation: -{observation}" 
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)



if __name__ == "__main__": 
    city = "上海" 
    weather = get_weather(city) 
    print(f'上海的天气: {weather}') 
    attractions = get_attraction(city, weather) 
    print(f'推荐景点如下: \n {attractions}')




