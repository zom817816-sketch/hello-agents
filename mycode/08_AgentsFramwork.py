# 定义全局状态 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 一个共享的数据结构，它在图的每个节点之间传递，作为工作流的持久化上下文
class SearchState(TypedDict): 
    messages: Annotated[list, add_messages] 
    user_query: str # 经过LLM理解后的用户需求总结 
    search_query: str # 优化后用于Tavily API的搜索查询 
    search_results: str # Tavily搜索返回的结果
    final_answer: str # 最终生成的答案 
    step: str # 标记当前步骤 

# 定义工作流节点 
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

# 加载.env中的环境变量 
load_dotenv() 

# 初始化模型 
# 使用LLM实例来驱动所有的节点的智能
llm = ChatOpenAI(
    model=os.getenv("MODEL"), 
    api_key=os.getenv("API_KEY"), 
    base_url=os.getenv("BASE_URL"), 
    temperature=0.7
) 

# 初始化Tavily客户端 
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 理解与查询节点 
def understand_query_node(state: SearchState) -> dict: 
    """ 步骤1：理解用户查询并生成搜索关键词 """
    user_message = state["messages"][-1].content

    understand_prompt = f""" 分析用户的查询： "{user_message}"
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""
    response = llm.invoke([SystemMessage(content=understand_prompt)]) 
    response_content = response.content

    # 解析LLM的输出，提取搜索关键词 
    if "搜索词：" in response_content:
        search_query = response_content.split("搜索词：")[1].strip() 
    
    return {
        "user_query": response_content, 
        "search_query": search_query, 
        "step": "understood", 
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")]
    }

# 搜索节点 
def tavily_search_node(state: SearchState) -> dict: 
    """步骤2：使用Tavily API进行真实搜索""" 
    search_query = state["search_query"]
    try: 
        print(f"🔍 正在搜索：{search_query}")
        response = tavily_client.search(
            query=search_query, 
            search_depth='basic', 
            max_results=5, 
            include_answer=True
        )
        # 处理和格式化搜索结果 
        search_results = "..." # 格式化后的结果字符串 

        return {
            "search_results": search_results, 
            "step": "searched", 
            "messages": [AIMessage(content="✅ 搜索完成！正在整理答案。。。")]
        }
    except Exception as e: 
        # 处理错误 
        return {
            "search_results": f"搜索失败：{e}",
            "step": "searched",
            "messages": [AIMessage(content="❌ 搜索遇到问题。。。")]
        }

# 回答节点 
def generate_answer_node(state: SearchState) -> dict: 
    """步骤3：基于搜索结果生成最终答案"""
    if state["step"] == "search_failed": 
        # 如果搜索失败，执行回退策略，基于LLM自身知识回答
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户问题：\n用户问题：{state["user_query"]}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else: 
        # 搜索成功，根据搜索结果生成答案 
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
        用户问题：{state["user_query"]}
        搜索结果：{state["search_results"]}
        请综合搜索结果，提供准确、有用的回答..."""
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content, 
        "step": "completed", 
        "messages": [AIMessage(content=response.content)]
    }

# 构建图 
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

def create_search_assistant(): 
    workflow = StateGraph(SearchState)

    # 添加节点 
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 添加边 - 定义工作流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    # 编译图 
    memory = InMemorySaver() 
    app = workflow.compile(checkpointer=memory)
    return app

# 主程序入口
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 智能搜索助手启动")
    print("=" * 60)
    
    # 创建应用
    app = create_search_assistant()
    
    # 配置线程ID（用于会话管理）
    config = {"configurable": {"thread_id": "demo_session"}}
    
    # 示例查询
    test_queries = [
        "什么是LangGraph？它有什么特点？",
        "Python异步编程的最佳实践"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"📝 测试 {i}: {query}")
        print("=" * 60)
        
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": ""
        }
        
        # 运行工作流
        result = app.invoke(initial_state, config)
        
        # 输出结果
        print(f"\n✨ 最终答案：")
        print("-" * 60)
        print(result["final_answer"])
        print("-" * 60)
        
        print(f"\n📊 工作流步骤：{result['step']}")
        print(f"🔍 搜索关键词：{result['search_query']}")

# 流程图的设计的最大优势是高度的可控性与可预测性，可以精确的规划智能体的每一步行为；对循环原生支持，支持自我优化和回退
# 每个节点都是一个Python函数，高度模块化
# 局限性：需要更多的前期代码（状态；节点；边）；缺少对话智能体的动态的、“涌现的”交互；对开发者的全局判断要求高

