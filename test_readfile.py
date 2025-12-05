"""
简单示例: 使用 read_npz_file tool 读取 .npz 文件，并将分析结果提供给 agent
"""
import os
import numpy as np
from smolagents import CodeAgent, LiteLLMModel, tool

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


@tool
def read_npz_file(file_path: str) -> str:
    """
    读取.npz文件并返回数据分析结果。
    
    Args:
        file_path: npz文件的路径
        
    Returns:
        包含数据结构和统计信息的字符串
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        result = f"=== NPZ文件分析: {file_path} ===\n\n"
        result += f"包含的数组: {list(data.keys())}\n\n"
        
        for key in data.keys():
            arr = data[key]
            result += f"--- {key} ---\n"
            result += f"  形状 (shape): {arr.shape}\n"
            result += f"  数据类型 (dtype): {arr.dtype}\n"
            
            if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                result += f"  最小值: {np.min(arr):.6f}\n"
                result += f"  最大值: {np.max(arr):.6f}\n"
                result += f"  均值: {np.mean(arr):.6f}\n"
                result += f"  标准差: {np.std(arr):.6f}\n"
                
                # 显示部分数据预览
                if arr.ndim == 1:
                    preview = arr[:5] if len(arr) > 5 else arr
                    result += f"  数据预览 (前5个): {preview}\n"
                elif arr.ndim == 2:
                    result += f"  数据预览 (前3行, 前5列):\n"
                    rows = min(3, arr.shape[0])
                    cols = min(5, arr.shape[1])
                    for i in range(rows):
                        result += f"    行{i}: {arr[i, :cols]}\n"
            result += "\n"
        
        data.close()
        return result
        
    except FileNotFoundError:
        return f"错误: 文件 {file_path} 不存在"
    except Exception as e:
        return f"错误: 读取文件失败 - {str(e)}"


def main():
    # 创建 LLM 模型
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash-lite",
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=4096,
    )
    
    # 创建带有 read_npz_file 工具的 agent
    agent = CodeAgent(
        model=model,
        tools=[read_npz_file],
        max_steps=5,
        verbosity_level=2,
        add_base_tools=False,
    )
    
    # 定义任务：读取并分析 npz 文件
    npz_path = "utils/Dainarx_code/data_duffing/sample_0.npz"
    
    task = f"""
    请使用 read_npz_file 工具读取文件: {npz_path}
    
    读取后，请根据返回的数据分析结果，简要总结：
    1. 这个数据文件包含哪些数组？
    2. 每个数组的维度和大小是多少？
    3. 这个数据可能用于什么场景？
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
