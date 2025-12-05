"""
简洁版本: 预加载数据 + state 注入，无需定义 tool

设计思路:
1. 在 agent 外部预先加载数据
2. 通过 agent.python_executor.state 注入数据
3. agent 直接写代码分析，不需要任何工具
"""
import os
import numpy as np
from smolagents import CodeAgent, LiteLLMModel

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


def load_npz_to_dict(file_path: str) -> dict:
    """加载 npz 文件并返回字典"""
    data = np.load(file_path, allow_pickle=True)
    result = {key: data[key].copy() for key in data.keys()}
    data.close()
    return result


def get_data_info(data: dict) -> str:
    """获取数据结构信息的字符串描述"""
    lines = []
    for key, arr in data.items():
        lines.append(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")
    return "\n".join(lines)


def main():
    # ========== 1. 预加载数据 ==========
    npz_path = "utils/Dainarx_code/data_duffing/sample_train_0.npz"
    
    print(f"正在加载数据: {npz_path}")
    loaded_data = load_npz_to_dict(npz_path)
    data_info = get_data_info(loaded_data)
    print(f"数据结构:\n{data_info}\n")
    
    # ========== 2. 创建 Agent（无需 tools）==========
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash-lite",
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=4096,
    )
    
    agent = CodeAgent(
        model=model,
        tools=[],  # 不需要任何工具
        max_steps=10,
        verbosity_level=2,
        add_base_tools=False,
        additional_authorized_imports=[
            "numpy",
            "pandas",
            "matplotlib",
            "matplotlib.pyplot",
            "scipy",
            "scipy.stats",
        ],
    )
    
    # ========== 3. 通过 state 注入数据 ==========
    agent.python_executor.state["data"] = loaded_data
    
    # ========== 4. 定义任务 ==========
    task = f"""
数据已预加载到变量 `data` 中，你可以直接使用。

数据来源: {npz_path}
数据结构:
{data_info}

请写 Python 代码完成以下分析:
1. 计算每个数组的基本统计量 (min, max, mean, std)
2. 检查数据是否包含异常值 (如 NaN 或 Inf)
3. 如果是时间序列数据，分析其趋势特征
4. 给出你对这个数据集的理解和可能的应用场景

注意:
- 直接使用 `data['array_name']` 访问数据
- 可以使用 numpy 进行计算
- 请写完整的分析代码
"""
    
    print("=" * 60)
    print("任务:", task)
    print("=" * 60)
    
    # ========== 5. 运行 Agent ==========
    result = agent.run(task)
    
    print("\n" + "=" * 60)
    print("Agent 最终答案:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
