"""
test_managedagent.py - smolagents ManagedAgent 讲解与示例

==========================================
smolagents ManagedAgent 概念详解
==========================================

什么是 ManagedAgent？
-------------------
ManagedAgent 是 smolagents 框架中用于构建**多智能体系统 (Multi-Agent System)** 的核心组件。
它允许你创建一个层级化的智能体架构，其中：

1. **Manager Agent (管理者智能体)**: 负责接收用户任务并协调其他智能体
2. **Managed Agent (被管理智能体)**: 专门化的智能体，由管理者调用执行特定任务

为什么使用 ManagedAgent？
----------------------
- **专业化分工**: 每个智能体专注于特定领域的任务
- **独立的工具集**: 不同智能体可以拥有不同的工具
- **隔离的记忆**: 每个智能体有自己独立的上下文和记忆
- **更好的扩展性**: 容易添加新的专业智能体
- **更高效的执行**: 管理者可以并行调度多个智能体

架构示意图
--------
```
                    ┌─────────────────┐
                    │  Manager Agent  │
                    │   (管理者)        │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Managed Agent │ │ Managed Agent │ │ Managed Agent │
    │  (搜索专家)     │ │  (计算专家)     │ │  (分析专家)     │
    └───────────────┘ └───────────────┘ └───────────────┘
```

使用方式
-------
方式1: 使用 ManagedAgent 包装类 (显式包装)
```python
from smolagents import CodeAgent, ManagedAgent

web_agent = CodeAgent(tools=[WebSearchTool()], model=model)
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="web_search",
    description="执行网络搜索"
)
manager = CodeAgent(tools=[], model=model, managed_agents=[managed_web_agent])
```

方式2: 直接传递带 name 和 description 的智能体 (隐式包装)
```python
from smolagents import CodeAgent

web_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="web_search",  # 直接指定名称
    description="执行网络搜索"  # 直接指定描述
)
manager = CodeAgent(tools=[], model=model, managed_agents=[web_agent])
```

"""

import os
import sys

# 确保可以导入 smolagents
try:
    from smolagents import (
        CodeAgent,
        ToolCallingAgent,
        LiteLLMModel,
        InferenceClientModel,
        Tool,
        MultiStepAgent,  # 所有智能体的基类
    )
    # 注意: 在新版本 smolagents 中，ManagedAgent 是隐式的概念
    # 只需要给 agent 设置 name 和 description，然后传入 managed_agents 参数
    # ManagedAgentPromptTemplate 是用于格式化提示词的模板
    from smolagents import ManagedAgentPromptTemplate
    
    SMOLAGENTS_AVAILABLE = True
    print("✓ smolagents 模块加载成功")
    print("  当前版本使用隐式 ManagedAgent - 通过 name/description 属性标识被管理智能体")
except ImportError as e:
    SMOLAGENTS_AVAILABLE = False
    print(f"smolagents 未安装: {e}")
    print("请运行: pip install 'smolagents[toolkit]'")


# =============================================================================
# 示例 1: 基础的 ManagedAgent 设置 (概念演示)
# =============================================================================

def demo_basic_managed_agent_concept():
    """
    演示 ManagedAgent 的基本概念和设置方式
    
    这个示例展示了如何：
    1. 创建一个专门化的 "工作" 智能体
    2. 将其包装为 ManagedAgent
    3. 创建一个管理者智能体来协调工作
    """
    print("\n" + "="*60)
    print("示例 1: ManagedAgent 基础概念")
    print("="*60)
    
    if not SMOLAGENTS_AVAILABLE:
        print("跳过 - smolagents 未安装")
        return
    
    # 使用环境变量中的 API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        print("警告: 未设置 GEMINI_API_KEY 或 HF_TOKEN 环境变量")
        print("本示例将仅展示代码结构，不会实际运行")
        return
    
    # -----------------------------
    # 步骤 1: 创建 LLM 模型
    # -----------------------------
    # LiteLLMModel 支持多种后端 (OpenAI, Gemini, Anthropic 等)
    model = LiteLLMModel(
        model_id="gemini/gemini-flash-lite-latest",  # 使用 Gemini 模型
        api_key=api_key,
        max_completion_tokens=2048,
    )
    
    # -----------------------------
    # 步骤 2: 创建专门化的工作智能体
    # -----------------------------
    # 这个智能体将专门处理数学计算任务
    math_agent = CodeAgent(
        tools=[],  # 可以添加数学相关的工具
        model=model,
        name="math_expert",  # 重要: 给智能体命名
        description="数学计算专家，可以执行复杂的数学运算和推导",
        max_steps=5,
        additional_authorized_imports=["numpy", "math", "scipy"],
    )
    
    # -----------------------------
    # 步骤 3: 创建管理者智能体
    # -----------------------------
    # 管理者智能体通过 managed_agents 参数来管理其他智能体
    manager_agent = CodeAgent(
        tools=[],  # 管理者自己的工具
        model=model,
        managed_agents=[math_agent],  # 关键: 将工作智能体作为被管理智能体
        max_steps=10,
        additional_authorized_imports=["time"],
    )
    
    print("✓ 创建了管理者智能体和被管理智能体")
    print(f"  - 管理者: CodeAgent")
    print(f"  - 被管理: math_expert (CodeAgent)")
    
    # -----------------------------
    # 步骤 4: 运行多智能体系统
    # -----------------------------
    print("\n运行示例任务...")
    try:
        result = manager_agent.run(
            "请计算 fibonacci(10) 的值，并解释你是如何计算的"
        )
        print(f"\n结果: {result}")
    except Exception as e:
        print(f"执行出错: {e}")
        

# =============================================================================
# 示例 2: ManagedAgentPromptTemplate 详解
# =============================================================================

def demo_managed_agent_prompt_template():
    """
    演示 ManagedAgentPromptTemplate 的作用
    
    在 smolagents 中，ManagedAgentPromptTemplate 用于定义
    管理者智能体如何调用被管理智能体的提示词模板
    """
    print("\n" + "="*60)
    print("示例 2: ManagedAgentPromptTemplate 详解")
    print("="*60)
    
    if not SMOLAGENTS_AVAILABLE:
        print("跳过 - smolagents 未安装")
        return
    
    print("""
ManagedAgentPromptTemplate 是用于格式化管理者与被管理智能体交互的模板。

当管理者智能体需要调用被管理智能体时，会使用这个模板生成指令：

1. **智能体调用格式**: 
   管理者通过被管理智能体的 `name` 来调用它
   
2. **任务描述传递**: 
   使用被管理智能体的 `description` 来理解其能力

3. **隐式包装方式** (当前版本推荐):
   ```python
   # 不需要显式的 ManagedAgent 类
   # 直接在 Agent 初始化时设置 name 和 description
   worker = CodeAgent(
       tools=[...],
       model=model,
       name="worker_name",        # 关键: 设置名称
       description="工作描述..."   # 关键: 设置描述
   )
   
   manager = CodeAgent(
       tools=[],
       model=model,
       managed_agents=[worker]   # 直接传入
   )
   ```

4. **在 CodeAgent 中调用被管理智能体**:
   ```python
   # 管理者生成的代码可以这样调用被管理智能体:
   result = worker_name("请执行某个任务")
   ```
""")
    
    print("✓ 当前版本使用隐式 ManagedAgent 机制")


# =============================================================================
# 示例 3: 多个 Managed Agents 协作
# =============================================================================

def demo_multiple_managed_agents():
    """
    演示多个被管理智能体协同工作
    
    场景: 
    - 管理者智能体接收复杂任务
    - 搜索智能体负责信息检索
    - 分析智能体负责数据分析
    - 写作智能体负责内容生成
    """
    print("\n" + "="*60)
    print("示例 3: 多个 Managed Agents 协作")
    print("="*60)
    
    if not SMOLAGENTS_AVAILABLE:
        print("跳过 - smolagents 未安装")
        return
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        print("跳过 - 未设置 API key")
        return
    
    model = LiteLLMModel(
        model_id="gemini/gemini-flash-lite-latest",
        api_key=api_key,
        max_completion_tokens=4096,
    )
    
    # -----------------------------
    # 创建多个专门化智能体
    # -----------------------------
    
    # 智能体 1: 数据处理专家
    data_agent = CodeAgent(
        tools=[],
        model=model,
        name="data_processor",
        description="数据处理专家，擅长数据清洗、转换和统计分析。给它数据处理任务。",
        max_steps=5,
        additional_authorized_imports=["numpy", "pandas"],
    )
    
    # 智能体 2: 可视化专家  
    viz_agent = CodeAgent(
        tools=[],
        model=model,
        name="visualization_expert",
        description="可视化专家，擅长创建图表和数据可视化。给它绘图任务。",
        max_steps=5,
        additional_authorized_imports=["matplotlib.pyplot"],
    )
    
    # 智能体 3: 报告生成专家
    report_agent = ToolCallingAgent(
        tools=[],
        model=model,
        name="report_writer",
        description="报告撰写专家，擅长总结信息并生成结构化报告。给它写作任务。",
        max_steps=5,
    )
    
    # -----------------------------
    # 创建管理者智能体
    # -----------------------------
    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[data_agent, viz_agent, report_agent],
        max_steps=15,
        additional_authorized_imports=["time", "json"],
    )
    
    print("✓ 创建了多智能体协作系统")
    print(f"  管理者: CodeAgent")
    print(f"  被管理智能体:")
    print(f"    - data_processor (CodeAgent)")
    print(f"    - visualization_expert (CodeAgent)")  
    print(f"    - report_writer (ToolCallingAgent)")
    
    # 展示如何调用
    print("\n管理者智能体可以这样调用被管理智能体:")
    print("```python")
    print('# 在管理者的代码执行环境中:')
    print('result = data_processor("请分析这组数据: [1,2,3,4,5]")')
    print('```')


# =============================================================================
# 示例 4: 自定义工具 + Managed Agent
# =============================================================================

def demo_custom_tool_with_managed_agent():
    """
    演示如何给 Managed Agent 配备自定义工具
    """
    print("\n" + "="*60)
    print("示例 4: 自定义工具与 Managed Agent")
    print("="*60)
    
    if not SMOLAGENTS_AVAILABLE:
        print("跳过 - smolagents 未安装")
        return
    
    # -----------------------------
    # 定义自定义工具
    # -----------------------------
    class CalculatorTool(Tool):
        """简单的计算器工具"""
        name = "calculator"
        description = "执行基本数学运算。输入: 数学表达式字符串"
        inputs = {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，如 '2+3*4'"
            }
        }
        output_type = "string"
        
        def forward(self, expression: str) -> str:
            try:
                # 安全地评估数学表达式
                result = eval(expression, {"__builtins__": {}}, {"abs": abs, "pow": pow})
                return f"计算结果: {expression} = {result}"
            except Exception as e:
                return f"计算错误: {e}"
    
    class UnitConverterTool(Tool):
        """单位转换工具"""
        name = "unit_converter"
        description = "进行单位转换。支持长度(m/km/mile)和温度(C/F/K)转换"
        inputs = {
            "value": {"type": "number", "description": "要转换的数值"},
            "from_unit": {"type": "string", "description": "源单位"},
            "to_unit": {"type": "string", "description": "目标单位"},
        }
        output_type = "string"
        
        def forward(self, value: float, from_unit: str, to_unit: str) -> str:
            conversions = {
                ("m", "km"): lambda x: x / 1000,
                ("km", "m"): lambda x: x * 1000,
                ("C", "F"): lambda x: x * 9/5 + 32,
                ("F", "C"): lambda x: (x - 32) * 5/9,
            }
            key = (from_unit, to_unit)
            if key in conversions:
                result = conversions[key](value)
                return f"{value} {from_unit} = {result:.2f} {to_unit}"
            return f"不支持从 {from_unit} 到 {to_unit} 的转换"
    
    print("✓ 定义了自定义工具:")
    print("  - CalculatorTool: 数学计算")
    print("  - UnitConverterTool: 单位转换")
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        print("\n跳过运行 - 未设置 API key")
        return
    
    model = LiteLLMModel(
        model_id="gemini/gemini-flash-lite-latest",
        api_key=api_key,
    )
    
    # 创建带有自定义工具的工作智能体
    math_worker = ToolCallingAgent(
        tools=[CalculatorTool(), UnitConverterTool()],
        model=model,
        name="math_worker",
        description="数学助手，可以进行计算和单位转换。给它数学问题。",
        max_steps=5,
    )
    
    # 创建管理者
    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[math_worker],
        max_steps=10,
    )
    
    print("\n✓ 创建了配备自定义工具的多智能体系统")
    

# =============================================================================
# 示例 5: 实际项目中的应用模式
# =============================================================================

def demo_real_world_pattern():
    """
    展示在实际项目中使用 ManagedAgent 的模式
    
    参考 run_llm_ha_beta.py 中的实现方式
    """
    print("\n" + "="*60)
    print("示例 5: 实际项目应用模式 (参考 run_llm_ha_beta.py)")
    print("="*60)
    
    print("""
在实际项目中 (如本项目的 run_llm_ha_beta.py)，ManagedAgent 的使用模式如下:

1. **工厂函数创建智能体**:
   ```python
   def get_managed_agents_list(managed_agents_list, model_id):
       managed_agents = []
       for agent_name in managed_agents_list:
           model = LiteLLMModel(model_id=model_id, ...)
           managed_agent = CodeAgent(
               tools=[],
               model=model,
               name=agent_name,
               description=f"I am a managed agent with name {agent_name}",
               max_steps=80,
           )
           managed_agents.append(managed_agent)
       return managed_agents
   ```

2. **在主智能体中注入被管理智能体**:
   ```python
   manager_agent = CodeAgent(
       model=model,
       tools=[...],
       managed_agents=get_managed_agents_list(['math_expert', 'search_agent'], model_id),
   )
   ```

3. **智能体间的数据共享** (高级用法):
   ```python
   # 在主智能体上添加自定义属性
   manager_agent.shared_data = {"key": "value"}
   
   # 工具可以通过 worker_agent 引用访问
   for tool in manager_agent.tools.values():
       tool.worker_agent = manager_agent
   ```

4. **动态任务分配**:
   管理者智能体会自动根据任务描述和被管理智能体的描述来决定
   将任务委派给哪个智能体。
""")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """运行所有示例"""
    print("="*60)
    print("smolagents ManagedAgent 功能讲解")
    print("="*60)
    
    # 检查依赖
    print("\n检查依赖...")
    if SMOLAGENTS_AVAILABLE:
        print("✓ smolagents 已安装")
    else:
        print("✗ smolagents 未安装")
        print("  运行: pip install 'smolagents[toolkit]'")
    
    # 检查 API key
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Load from .env file in the current directory
    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("HF_TOKEN"))
    if has_api_key:
        print("✓ API key 已配置")
    else:
        print("⚠ 未配置 API key (GEMINI_API_KEY 或 HF_TOKEN)")
        print("  部分示例将只展示代码结构")
    
    # 运行示例
    demo_basic_managed_agent_concept()
    # demo_managed_agent_prompt_template()
    # demo_multiple_managed_agents()
    # demo_custom_tool_with_managed_agent()
    # demo_real_world_pattern()
    
    # 总结
    print("\n" + "="*60)
    print("总结: ManagedAgent 的核心要点")
    print("="*60)
    print("""
1. **创建专门化智能体**: 
   为不同任务创建专门的智能体，配备相应的工具

2. **设置 name 和 description**: 
   这是管理者智能体识别和调用被管理智能体的关键

3. **通过 managed_agents 参数注入**:
   manager = CodeAgent(managed_agents=[agent1, agent2, ...])

4. **管理者自动协调**:
   管理者智能体会根据任务需求自动决定调用哪个被管理智能体

5. **独立的执行环境**:
   每个被管理智能体有自己独立的工具集、记忆和执行上下文
""")


if __name__ == "__main__":
    main()

