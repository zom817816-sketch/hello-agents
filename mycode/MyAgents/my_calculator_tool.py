import ast 
import operator
import math 
from .my_tools import ToolRegistry

"""
安全的数学计算工具模块

该模块提供了一个安全的数学表达式计算器，通过解析抽象语法树(AST)来计算数学表达式，
避免了直接使用eval()函数的安全风险。

主要功能：
- 支持基本运算：加减乘除、幂运算、取模、整除
- 支持常用数学函数：sqrt、sin、cos、tan、log、exp、abs等
- 支持数学常量：pi、e
- 安全的表达式求值，防止代码注入攻击

使用示例：
    from my_calculator_tool import my_calculate
    result = my_calculate("3+5*sqrt(16)")  # 返回 "23.0"
"""

def my_calculate(expression: str) -> str: 
    """ 
    安全的数学计算函数，支持基本运算和数学函数
    
    该函数通过解析抽象语法树(AST)来计算数学表达式，避免了直接使用eval()的安全风险。
    这种方法只允许预定义的运算符和函数，有效防止了代码注入攻击。

    Args:
        expression (str): 要计算的数学表达式字符串
                        支持的运算符: +, -, *, /, **, //, %
                        支持的函数: sqrt, sin, cos, tan, log, exp, abs
                        支持的常量: pi, e
                        示例: "3+5*sqrt(16)", "2**3+sin(0)", "pi*2"

    Returns:
        str: 计算结果字符串，或错误信息
              示例: "23.0", "8.0", "计算失败: 语法错误"

    Examples:
        >>> my_calculate("3+5*sqrt(16)")
        "23.0"
        >>> my_calculate("2**3+sin(0)")
        "8.0"
        >>> my_calculate("pi*2")
        "6.283185307179586"
        >>> my_calculate("")
        "计算表达式不能为空"
    """ 
    # 检查输入是否为空
    if not expression.strip(): 
        return "计算表达式不能为空"
    
    # 定义支持的基本运算符映射
    # 将AST节点类型映射到对应的Python运算符函数
    operators = {
        ast.Add: operator.add,        # 加法: +
        ast.Sub: operator.sub,        # 减法: -
        ast.Mult: operator.mul,       # 乘法: *
        ast.Div: operator.truediv,    # 除法: /
        ast.Pow: operator.pow,        # 幂运算: **
        ast.FloorDiv: operator.floordiv,  # 整除: //
        ast.Mod: operator.mod         # 取模: %
    }

    # 定义支持的数学函数和常量
    # 将函数/常量名称映射到实际的Python函数/值
    functions = {
        # 数学函数
        "sqrt": math.sqrt,    # 平方根
        "sin": math.sin,      # 正弦函数
        "cos": math.cos,      # 余弦函数
        "tan": math.tan,      # 正切函数
        "log": math.log,      # 自然对数
        "exp": math.exp,      # 指数函数
        "abs": abs,           # 绝对值
        # 数学常量
        "pi": math.pi,        # 圆周率 π
        "e": math.e           # 自然常数 e
    }

    try: 
        # 将表达式字符串解析为抽象语法树(AST)
        # mode="eval" 表示解析为表达式而非语句，提高了安全性
        node = ast.parse(expression, mode="eval")
        
        # 递归求值AST节点，从表达式体的根节点开始计算
        result = _eval_node(node.body, operators, functions)
        
        # 将结果转换为字符串返回
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as e:
        # 捕获常见的计算错误并返回友好的错误信息
        return f"计算失败: {str(e)}"
    
def _eval_node(node, operators, functions): 
    """ 
    递归求值AST节点，实现安全的表达式计算
    
    该函数使用递归下降策略遍历抽象语法树，从叶子节点开始计算，
    逐步组合得到最终结果。这是整个计算器的核心引擎。

    Args:
        node: AST节点，可以是常量、运算、函数调用等
        operators (dict): 运算符映射字典，{AST类型: 运算符函数}
        functions (dict): 函数和常量映射字典，{名称: 函数/值}

    Returns:
        计算结果，类型可能是数字、字符串等

    Raises:
        ValueError: 当遇到不支持的运算符、函数或变量时
        ZeroDivisionError: 当除数为零时

    Examples:
        >>> # 表达式 "3+5*sqrt(16)" 的AST求值过程:
        >>> # 1. 递归计算左子树: ast.Constant(3) -> 3
        >>> # 2. 递归计算右子树: 
        >>> #    - ast.Constant(5) -> 5
        >>> #    - ast.Call(sqrt, [ast.Constant(16)]) -> math.sqrt(16) -> 4.0
        >>> #    - ast.BinOp(5, *, 4.0) -> 5 * 4.0 -> 20.0
        >>> # 3. 执行顶层运算: 3 + 20.0 -> 23.0
    """
    
    # 处理常量节点（数字、字符串等字面量）
    if isinstance(node, ast.Constant): 
        # 直接返回常量的值
        # 例如: 42 -> 42, 3.14 -> 3.14
        return node.value
        
    # 处理二元运算节点（+、-、*、/、**、//、%）
    elif isinstance(node, ast.BinOp): 
        # 递归计算左操作数
        left = _eval_node(node.left, operators, functions)
        
        # 递归计算右操作数
        right = _eval_node(node.right, operators, functions)
        
        # 获取运算符对应的函数
        op = operators.get(type(node.op))
        
        # 检查运算符是否支持
        if op is None: 
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        
        # 执行运算并返回结果
        # 例如: operator.add(3, 5) -> 8
        return op(left, right)
        
    # 处理函数调用节点（sqrt、sin、cos等）
    elif isinstance(node, ast.Call): 
        # 获取函数名
        func_name = node.func.id 
        
        # 检查函数是否在允许的函数列表中
        if func_name in functions: 
            # 递归计算所有参数
            # 注意: 这里修复了原代码中的变量名冲突问题
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            
            # 调用函数并返回结果
            # 例如: functions["sqrt"]([16.0]) -> math.sqrt(16.0) -> 4.0
            return functions[func_name](*args)
        else: 
            # 函数不在允许列表中，拒绝执行
            raise ValueError(f"不支持的函数: {func_name}")
            
    # 处理变量名节点（pi、e等常量）
    elif isinstance(node, ast.Name): 
        # 检查变量名是否在允许的常量列表中
        if node.id in functions: 
            # 返回常量的值
            # 例如: node.id="pi" -> math.pi -> 3.14159...
            return functions[node.id]
        else: 
            # 变量名不在允许列表中，拒绝执行
            raise ValueError(f"不支持的变量: {node.id}")
            
    # 处理一元运算节点（负号-、正号+）
    elif isinstance(node, ast.UnaryOp):
        # 递归计算操作数
        operand = _eval_node(node.operand, operators, functions)
        
        # 处理负号
        if isinstance(node.op, ast.USub):
            # 例如: -5 -> -5, -(3+2) -> -5
            return -operand
        # 处理正号
        elif isinstance(node.op, ast.UAdd):
            # 例如: +5 -> 5, +(3+2) -> 5
            return +operand
            
    # 处理表达式根节点
    elif isinstance(node, ast.Expression):
        # 递归求值表达式体
        # 这是AST解析的入口点
        return _eval_node(node.body, operators, functions)
        
def create_calculator_registry(): 
    """创建包含计算器的工具注册表
    
    该函数创建一个工具注册表实例，并将计算器函数注册到其中。
    这样AI Agent就可以通过工具调用机制来使用数学计算功能。

    Returns:
        ToolRegistry: 包含已注册计算器工具的工具注册表对象

    Examples:
        >>> registry = create_calculator_registry()
        >>> # Agent可以通过以下方式调用计算器:
        >>> # result = registry.execute_tool("my_calculator", "3+5*sqrt(16)")
        >>> # result -> "23.0"
    """ 
    # 创建工具注册表实例
    registry = ToolRegistry() 

    # 注册计算器函数到工具注册表
    registry.register_function( 
        name="my_calculator",              # 工具名称，Agent通过此名称调用
        description="简单的数学计算工具，支持基本运算(+,-,*,/,**,//,%)和数学函数(sqrt,sin,cos,tan,log,exp,abs)", 
        func=my_calculate                   # 实际执行的函数
    ) 

    return registry 