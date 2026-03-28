import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


def call_llm(prompt: str, model: str = "doubao-seed-2-0-mini-260215") -> str:
    """调用OpenAI API获取回复"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"错误: {str(e)}"


# 任务定义
TASKS = [
    {
        "name": "数学推理 - 年龄问题",
        "question": "小明比小红大5岁，3年后小明的年龄是小红的2倍。请问小明现在多少岁？"
    },
    {
        "name": "逻辑推理 - 水果重量",
        "question": "一个苹果和一个香蕉共重300克，一个苹果和一个橙子共重400克，一个香蕉和一个橙子共重350克。请问苹果、香蕉、橙子各重多少克？"
    },
    {
        "name": "常识推理 - 时间安排",
        "question": "小王需要在上午完成三件事：洗衣服需要40分钟，做饭需要30分钟，打扫房间需要20分钟。他只有一台洗衣机，但做饭和打扫可以同时进行。请问最少需要多少分钟完成所有事情？"
    }
]


# 提示策略

def zero_shot_prompt(question: str) -> str:
    """Zero-shot: 直接提问，不给示例"""
    return f"请回答以下问题：\n\n{question}"


def few_shot_prompt(question: str) -> str:
    """Few-shot: 提供几个示例"""
    examples = """
示例1：
问题：小李比小张大3岁，2年后小李的年龄是小张的1.5倍。请问小李现在多少岁？
解答：设小张现在x岁，则小李现在x+3岁。
2年后：小李 = x+5，小张 = x+2
根据题意：x+5 = 1.5(x+2)
解得：x = 4
所以小李现在7岁。

示例2：
问题：一个篮球和一个足球共200元，篮球比足球贵40元。请问篮球多少钱？
解答：设足球x元，则篮球x+40元。
x + (x+40) = 200
2x = 160
x = 80
所以篮球120元。

现在请回答：
"""
    return examples + question


def cot_prompt(question: str) -> str:
    """Chain-of-Thought: 要求模型逐步思考"""
    return f"请回答以下问题。请一步一步地思考，展示你的推理过程，最后给出答案。\n\n问题：{question}\n\n解答："


def few_shot_cot_prompt(question: str) -> str:
    """Few-shot + CoT: 结合两种策略"""
    examples = """
请按照以下示例的格式，逐步推理并回答问题。

示例1：
问题：小李比小张大3岁，2年后小李的年龄是小张的1.5倍。请问小李现在多少岁？
解答：
步骤1：设未知数。设小张现在x岁，则小李现在x+3岁。
步骤2：表示2年后的年龄。2年后小李x+5岁，小张x+2岁。
步骤3：列方程。根据题意：x+5 = 1.5(x+2)
步骤4：解方程。x+5 = 1.5x+3，得0.5x=2，x=4
步骤5：得出答案。小李现在x+3=7岁。
答案：7岁

现在请回答：
"""
    return examples + question

def analyze_task(task_index: int = 0):
    """测试不同策略的输出效果"""
    
    task = TASKS[task_index]
    strategies = [
        ("Zero-shot", zero_shot_prompt),
        ("Few-shot", few_shot_prompt),
        ("CoT", cot_prompt),
        ("Few-shot + CoT", few_shot_cot_prompt)
    ]
    
    print("-" * 80)
    print(f"详细分析: {task['name']}")
    print("-" * 80)
    print(f"\n问题: {task['question']}\n")
    
    results = {}
    
    for strategy_name, strategy_func in strategies:
        print(f"\n{'='*60}")
        print(f"策略: {strategy_name}")
        print(f"{'='*60}")
        
        prompt = strategy_func(task['question'])
        response = call_llm(prompt)
        results[strategy_name] = response
        
        print(response)
        print()
        time.sleep(1)



if __name__ == "__main__":
        for i, task in enumerate(TASKS):
            print(f"{i}. {task['name']}")
            analyze_task(i)



