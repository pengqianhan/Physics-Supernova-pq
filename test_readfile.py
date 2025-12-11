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


def main():
    # 创建 LLM 模型
    model = LiteLLMModel(
        model_id="gemini/gemini-flash-lite-latest",
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=4096,
    )
    
    # 数据文件的绝对路径
    data_file_path = os.path.abspath("utils/Dainarx_code/data_duffing/sample_train_0.npz")
    
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
   import numpy as np
   data = np.load(DATA_FILE_PATH, allow_pickle=True)

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
