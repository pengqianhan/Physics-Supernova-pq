"""
测试 run_llm_ha_beta.py 中 get_managed_agents_list 定义的 managed_agent
验证其可以正常加载数据和运行
"""
import os
import sys
import numpy as np

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✓ dotenv loaded successfully")
except ImportError:
    print("⚠ dotenv not available, continuing without it")


def test_data_file_exists():
    """测试数据文件是否存在"""
    print("\n" + "=" * 60)
    print("测试 1: 验证数据文件存在")
    print("=" * 60)
    
    # 使用与 get_managed_agents_list 相同的路径构建方式
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trace_file_path = os.path.join(script_dir, "utils", "Dainarx_code", "data_duffing", "sample_train_0.npz")
    
    if os.path.exists(trace_file_path):
        print(f"✓ 数据文件存在: {trace_file_path}")
        
        # 加载并检查数据结构
        data = np.load(trace_file_path, allow_pickle=True)
        print(f"  数组名称: {list(data.keys())}")
        for key in data.keys():
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        return True
    else:
        print(f"✗ 数据文件不存在: {trace_file_path}")
        return False


def test_get_managed_agents_list_creation():
    """测试 get_managed_agents_list 函数是否可以正常创建 managed agents"""
    print("\n" + "=" * 60)
    print("测试 2: 测试 get_managed_agents_list 函数创建 managed agents")
    print("=" * 60)
    
    from run_llm_ha_beta import get_managed_agents_list
    
    # 测试空输入
    result_empty = get_managed_agents_list(None, None)
    assert result_empty == [], f"期望空列表，实际得到: {result_empty}"
    print("✓ 空输入返回空列表")
    
    # 测试创建单个 agent
    managed_agents = get_managed_agents_list(
        managed_agents_list=['data_analysis_expert'],
        managed_agents_list_model_id="gemini/gemini-flash-lite-latest"
    )
    
    assert len(managed_agents) == 1, f"期望 1 个 agent，实际得到 {len(managed_agents)} 个"
    print(f"✓ 成功创建 1 个 managed agent")
    
    agent = managed_agents[0]
    print(f"  Agent 名称: {agent.name}")
    print(f"  Agent 描述: {agent.description[:100]}...")
    
    # 验证 DATA_FILE_PATH 是否正确注入
    data_file_path = agent.python_executor.state.get("DATA_FILE_PATH")
    assert data_file_path is not None, "DATA_FILE_PATH 未注入到 agent 状态"
    print(f"✓ DATA_FILE_PATH 已注入: {data_file_path}")
    
    # 验证文件路径是否有效
    assert os.path.exists(data_file_path), f"注入的文件路径不存在: {data_file_path}"
    print(f"✓ 注入的文件路径有效")
    
    return managed_agents


def test_managed_agent_can_execute_code():
    """测试 managed agent 的 Python 执行器是否可以运行代码"""
    print("\n" + "=" * 60)
    print("测试 3: 测试 managed agent 的 Python 执行器")
    print("=" * 60)
    
    from run_llm_ha_beta import get_managed_agents_list
    
    managed_agents = get_managed_agents_list(
        managed_agents_list=['data_analysis_expert'],
        managed_agents_list_model_id="gemini/gemini-flash-lite-latest"
    )
    
    agent = managed_agents[0]
    executor = agent.python_executor
    
    # 测试简单代码执行
    simple_code = """
result = 1 + 1
"""
    output = executor(simple_code)
    assert output.output == 2, f"简单计算失败，期望 2，得到 {output.output}"
    print("✓ 简单计算: 1 + 1 = 2")
    
    # 测试 numpy 导入和使用
    numpy_code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.mean(arr)
"""
    output = executor(numpy_code)
    assert output.output == 3.0, f"NumPy 计算失败，期望 3.0，得到 {output.output}"
    print("✓ NumPy 计算: mean([1,2,3,4,5]) = 3.0")
    
    # 测试访问 DATA_FILE_PATH
    # 注意: smolagents 的 LocalPythonExecutor 对内置函数有限制
    # 不能直接使用 list(), len() 等内置函数，需要用 numpy 方式
    access_data_code = """
import numpy as np
data = np.load(DATA_FILE_PATH, allow_pickle=True)
# 直接获取 state 数组的 shape，验证文件可读取
state = data['state']
result = state.shape[0]  # 返回第一维大小
"""
    output = executor(access_data_code)
    assert output.output >= 1, f"无法访问数据文件，得到: {output.output}"
    print(f"✓ 成功读取数据文件，state 数组第一维大小: {output.output}")
    
    # 测试数据分析
    # 注意: smolagents 限制了 float(), int() 等内置函数的使用
    # 使用 numpy 的 item() 方法来获取标量值
    analysis_code = """
import numpy as np
data = np.load(DATA_FILE_PATH, allow_pickle=True)
state = data['state']
result = {
    'shape': state.shape,
    'min': np.min(state).item(),
    'max': np.max(state).item(),
    'mean': np.mean(state).item(),
    'std': np.std(state).item()
}
"""
    output = executor(analysis_code)
    result = output.output
    print(f"✓ 数据分析结果:")
    print(f"  - Shape: {result['shape']}")
    print(f"  - Min: {result['min']:.4f}")
    print(f"  - Max: {result['max']:.4f}")
    print(f"  - Mean: {result['mean']:.4f}")
    print(f"  - Std: {result['std']:.4f}")
    
    return True


def test_managed_agent_run_with_llm():
    """测试 managed agent 通过 LLM 运行任务（需要 API key）"""
    print("\n" + "=" * 60)
    print("测试 4: 测试 managed agent 通过 LLM 运行任务")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠ 跳过此测试: GEMINI_API_KEY 未设置")
        return None
    
    from run_llm_ha_beta import get_managed_agents_list
    
    managed_agents = get_managed_agents_list(
        managed_agents_list=['data_analysis_expert'],
        managed_agents_list_model_id="gemini/gemini-flash-lite-latest"
    )
    
    agent = managed_agents[0]
    
    # 简单任务：读取数据并分析
    task = """
请读取 DATA_FILE_PATH 中的 .npz 数据文件，分析其中 'state' 数组的基本统计信息。

步骤：
1. 使用 np.load(DATA_FILE_PATH, allow_pickle=True) 读取数据
2. 获取 'state' 数组
3. 计算并返回：shape, min, max, mean, std

请直接返回分析结果。
"""
    
    print("发送任务给 managed agent...")
    print(f"任务: {task[:100]}...")
    
    try:
        result = agent.run(task)
        print(f"\n✓ Agent 运行成功!")
        print(f"Agent 返回结果:\n{result[:500] if len(str(result)) > 500 else result}")
        return True
    except Exception as e:
        print(f"✗ Agent 运行失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("测试 get_managed_agents_list 定义的 managed_agent")
    print("=" * 60)
    
    all_passed = True
    
    # 测试 1: 数据文件存在性
    if not test_data_file_exists():
        print("\n✗ 测试 1 失败: 数据文件不存在")
        all_passed = False
    
    # 测试 2: get_managed_agents_list 函数
    try:
        test_get_managed_agents_list_creation()
    except Exception as e:
        print(f"\n✗ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 测试 3: Python 执行器
    try:
        test_managed_agent_can_execute_code()
    except Exception as e:
        print(f"\n✗ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 测试 4: LLM 运行（可选，需要 API key）
    try:
        result = test_managed_agent_run_with_llm()
        if result is False:
            all_passed = False
    except Exception as e:
        print(f"\n✗ 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    if all_passed:
        print("✓ 所有测试通过！managed_agent 可以正常加载数据和运行")
    else:
        print("✗ 部分测试失败，请检查上述错误信息")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
