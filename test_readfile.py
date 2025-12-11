"""
简单示例: 让 CodeAgent 直接写代码读取和分析 .npz 文件（不使用工具）
"""
import os
import numpy as np
from smolagents import CodeAgent, LiteLLMModel

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


def verify_working_directory():
    """验证主脚本和 CodeAgent 执行器的工作目录是否一致"""
    from smolagents.local_python_executor import LocalPythonExecutor
    
    # 创建执行器
    executor = LocalPythonExecutor(additional_authorized_imports=["os"])
    
    # 测试代码：获取当前工作目录
    code = """
import os
cwd = os.getcwd()
"""
    output = executor(code)
    executor_cwd = output.output
    main_cwd = os.getcwd()
    
    print("=" * 60)
    print("工作目录验证:")
    print(f"  主脚本工作目录:     {main_cwd}")
    print(f"  CodeAgent执行器目录: {executor_cwd}")
    print(f"  两者是否相同: {executor_cwd == main_cwd}")
    print("=" * 60)
    
    return executor_cwd == main_cwd


def main():
    # 验证工作目录
    if not verify_working_directory():
        print("警告: 工作目录不一致，可能会导致文件路径问题！")
    
    # 创建 LLM 模型
    model = LiteLLMModel(
        model_id="gemini/gemini-flash-lite-latest",
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=4096,
    )
    
    # 数据文件路径
    data_file_path = os.path.join(os.path.dirname(__file__), "utils", "Dainarx_code", "data_duffing", "sample_train_0.npz")
    # data_file_path = "utils/Dainarx_code/data_duffing/sample_train_0.npz"
    
    # 验证文件是否存在
    if os.path.exists(data_file_path):
        print(f"✓ 数据文件存在: {data_file_path}")
    else:
        print(f"✗ 数据文件不存在: {data_file_path}")
        print(f"  当前工作目录: {os.getcwd()}")
        return
    
    # 创建 CodeAgent（不使用任何工具，让 agent 直接写代码）
    agent = CodeAgent(
        model=model,
        tools=[],  # 不使用任何工具
        max_steps=10,
        verbosity_level=2,
        add_base_tools=False,
        additional_authorized_imports=["numpy", "os", "matplotlib", "matplotlib.pyplot"],
    )
    
    # 将数据文件路径注入到 agent 的状态中，让 agent 可以直接访问
    agent.python_executor.state["DATA_FILE_PATH"] = data_file_path
    
    # 定义任务：让 agent 直接写代码分析数据
    task = f"""
数据文件路径已存储在变量 DATA_FILE_PATH 中:
DATA_FILE_PATH = "{data_file_path}"

请直接写 Python 代码完成以下任务：

1. 使用 numpy 读取 npz 文件:
   ```python
   import numpy as np
   data = np.load(DATA_FILE_PATH, allow_pickle=True)
   ```

2. 分析数据结构:
   - 列出所有数组的名称
   - 打印每个数组的 shape 和 dtype

3. 计算基本统计量:
   - 每个数组的 min, max, mean, std

4. 根据分析结果，推测这个数据可能用于什么场景
"""
    
    print("=" * 60)
    print("任务:", task)
    print("=" * 60)
    
    # 运行 agent
    result = agent.run(task)
    
    print("\n" + "=" * 60)
    print("Agent 最终回答:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
