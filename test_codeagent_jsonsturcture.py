"""CodeAgent + LiteLLM 结构化输出示例"""
import json
import os
import numpy as np
from pydantic import BaseModel
import litellm
from smolagents import CodeAgent, LiteLLMModel

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# ========== Pydantic 模型 ==========
class ArrayInfo(BaseModel):
    """单个数组的信息"""
    name: str  # 数组名称（如 "state", "input"）
    shape: list[int]
    dtype: str
    min_value: float | None = None  # 空数组时为 None
    max_value: float | None = None  # 空数组时为 None


class DataAnalysisResult(BaseModel):
    """数据分析结果 - 使用 list 而非 dict 以兼容 Gemini"""
    source: str
    arrays: list[ArrayInfo]  # 改为 list 以兼容 Gemini JSON Schema
    total_elements: int
    summary: str


# ========== 工具函数 ==========
def load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k].copy() for k in data.keys()}


def get_data_info(data: dict) -> str:
    return "\n".join(f"  - {k}: shape={v.shape}, dtype={v.dtype}" for k, v in data.items())


def structured_completion(model_id: str, messages: list, response_model: type[BaseModel], api_key: str) -> BaseModel:
    """LiteLLM 结构化输出"""
    response = litellm.completion(
        model=model_id,
        messages=messages,
        api_key=api_key,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": True,
            },
        },
    )
    return response_model.model_validate(json.loads(response.choices[0].message.content))


def print_section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ========== 主流程 ==========
def main():
    npz_path = "data_all/ATVA/ball/sample_0.npz"
    api_key = os.environ.get("GEMINI_API_KEY")

    # 1. 加载数据
    data = load_npz(npz_path)
    data_info = get_data_info(data)
    print(f"数据: {npz_path}\n{data_info}")

    # 2. 创建 Agent
    agent = CodeAgent(
        model=LiteLLMModel(model_id="gemini/gemini-flash-lite-latest", api_key=api_key, max_completion_tokens=4096),
        tools=[],
        max_steps=5,
        verbosity_level=2,
        add_base_tools=False,
        additional_authorized_imports=["numpy", "pandas", "scipy", "scipy.stats"],
    )

    # 3. 注入数据
    agent.python_executor.state.update({"data": data, "npz_path": npz_path, "analysis_result": {}})

    # 4. 执行分析任务
    task = f"""
数据已加载到 `data`，路径在 `npz_path`。
{data_info}

请分析数据并存储到 `analysis_result`:
analysis_result = {{
    "source": npz_path,
    "arrays": [
        {{"name": "array_name", "shape": [...], "dtype": "...", "min_value": x_or_None, "max_value": y_or_None}}
        # 注意: 空数组的 min_value 和 max_value 设为 None
    ],
    "total_elements": N,
    "summary": "描述"
}}
"""
    print_section("阶段 1: CodeAgent 执行")
    agent.run(task)

    # 5. 提取结果
    result = agent.python_executor.state.get("analysis_result", {})
    print_section("Agent 结果")
    print(result)

    # 6. 结构化输出
    print_section("阶段 2: 结构化输出")

    # 尝试直接验证
    if result and "arrays" in result and isinstance(result["arrays"], list):
        try:
            output = DataAnalysisResult.model_validate(result)
            print("直接验证成功!")
            print(output.model_dump_json(indent=2))
            return output
        except Exception as e:
            print(f"验证失败: {e}，使用 LLM 格式化...")

    # LLM 格式化
    output = structured_completion(
        model_id="gemini/gemini-2.0-flash",
        messages=[
            {"role": "system", "content": "将数据分析结果转换为结构化格式。arrays 必须是列表，每个元素包含 name, shape, dtype, min_value, max_value。空数组的 min/max 值设为 null。"},
            {"role": "user", "content": f"数据源: {npz_path}\n结果: {result}\n信息: {data_info}"},
        ],
        response_model=DataAnalysisResult,
        api_key=api_key,
    )

    print_section("最终输出")
    print(output.model_dump_json(indent=2))
    return output


if __name__ == "__main__":
    main()
