from typing import Dict, List, Any
from pydantic import BaseModel

# Tool基类的抽象设计
class Tool(ABC): 
    """
    工具基类
    """
    def __init__(self, name: str, description: str): 
        self.name = name 
        self.description = description
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str: 
        """
        执行工具
        """
        pass 

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]: 
        """
        获取工具参数定义
        """
        pass

    def _to_openai_schema(self) -> Dict[str, Any]: 
        """
        转换为OpenAI function calling schema 格式
        用于 FunctionCallAgent，使工具能够被 OpenAI 原生 function calling 使用
        Returns:
            符合 OpenAI function calling 标准的 schema
        """
        parameters = self.get_parameters()

        # 构建properties 
        properties = {}
        required = []

        for param in parameters: 
            # 基础属性定义
            prop = {
                "type": param.type, 
                "description": param.description
            }

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if param.default: 
                prop["description"] = f"{param.description} (默认：{param.description})"
            
            # 如果是数组类型，添加items定义
            if param.type == "array": 
                prop["items"] = {"type": "string"} # 默认字符串数组
            
            properties[param.name] = prop

            # 收集必要参数
            if param.required: 
                required.append(param.name)
            
        return {
            "type": "function", 
            "function": {
                "name": self.name, 
                "description": self.description, 
                "parameters": {
                    "type": "object", 
                    "properties": properties, 
                    "required": required
                }
            }
        }


# ToolParameter参数定义系统
class ToolParameter(BaseModel): 
    """ 
    工具参数定义
    """
    name: str
    type: str 
    description: str
    required: bool = True
    default: Any = None 

# ToolRegistry注册表的实现
# 支持Tool对象注册和函数直接注册
class ToolRegistry: 
    """ 
    工具注册表
    """ 
    def __init__(self): 
        self._tools: Dict[str, Tool] = {}
        self._funcs: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, tool: Tool): 
        """
        注册工具对象
        """ 
        if tool.name in self._tools: 
            print(f"⚠️ 警告:工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def register_function(self, name: str, description: str, func: callable[[str], str]): 
        """ 
        直接注册函数为工具（简便方式）
        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符串结果
        """
        if name in self._funcs: 
            print(f"⚠️ 警告:工具 '{name}' 已存在，将被覆盖。")
        
        self._funcs[name] = {
            "description": description, 
            "func": func
        }
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def get_tools_description(self) -> str: 
        """
        获取所有有用工具的格式化描述字符串
        """
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values(): 
            descriptions.append(f"- {tool.name}: {tool.description}")
        
        # 函数工具描述
        for name, info in self._funcs.items(): 
            descriptions.append(f"- {name}: {info["description"]}")
        
        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        执行指定的工具
        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数
        Returns:
            工具执行结果
        """
        # 首先在Tool对象中查找
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            try:
                return tool.run({"input": tool_input})
            except Exception as e:
                return f"工具执行错误: {e}"
        
        # 然后在函数工具中查找
        elif tool_name in self._funcs:
            func_info = self._funcs[tool_name]
            func = func_info["func"]
            try:
                return func(tool_input)
            except Exception as e:
                return f"工具执行错误: {e}"
        
        # 未找到工具
        else:
            return f"未找到名为 '{tool_name}' 的工具"
