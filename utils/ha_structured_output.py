"""
Hybrid Automaton Structured Output Converter

两阶段架构的第二阶段：将 Agent 的自由文本输出转换为结构化 JSON

使用方法:
    from utils.ha_structured_output import convert_agent_result_to_ha

    # Agent 完成后
    result = agent.run(task, images=images)
    
    # 结构化转换
    ha_dict, success, message = convert_agent_result_to_ha(result, model_id="gemini/gemini-2.0-flash")
"""

import os
import json
import re
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 尝试导入 litellm
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    print("Warning: litellm not installed. Structured output conversion will not be available.")

# 导入现有的验证函数
try:
    from .ha_json_schema import validate_ha_with_schema
    from .ha_spec_validator import extract_and_fix_ha_spec
except ImportError:
    from ha_json_schema import validate_ha_with_schema
    from ha_spec_validator import extract_and_fix_ha_spec


# ==============================================================================
# Pydantic Models (Gemini 兼容版 - 无 anyOf/Optional)
# ==============================================================================

class HAModeSchema(BaseModel):
    """Hybrid Automaton 模态"""
    id: int = Field(..., ge=1, description="Mode ID (integer >= 1)")
    eq: str = Field(..., min_length=1, description="ODE equation, e.g., 'x1[1] = x2[0], x2[1] = -9.8'")


class HAEdgeSchema(BaseModel):
    """模态转换边"""
    direction: str = Field(..., description="Transition direction, e.g., '1 -> 2'")
    condition: str = Field(..., min_length=1, description="Guard condition, e.g., 'x1 <= 0'")
    reset_expr: str = Field(default="", description="Reset expression as string, e.g., 'x2=-0.9*x2[0]' or empty")


class HAAutomatonSchema(BaseModel):
    """Automaton 定义"""
    var: str = Field(..., description="State variables, e.g., 'x1' or 'x1, x2'")
    input: str = Field(default="", description="Input variables, e.g., 'u1' or ''")
    mode: list[HAModeSchema] = Field(..., min_length=1, description="List of modes")
    edge: list[HAEdgeSchema] = Field(default_factory=list, description="List of edges (can be empty)")


class HAConfigSchema(BaseModel):
    """配置参数"""
    dt: float = Field(..., gt=0, description="Time step, e.g., 0.001")
    total_time: float = Field(..., gt=0, description="Total simulation time, e.g., 10.0")
    order: int = Field(default=1, ge=1, description="ODE order: 1 for first-order ODE, 2 for second-order ODE")
    self_loop: bool = Field(default=False, description="Whether self-loop transitions exist")


class HASpecificationSchema(BaseModel):
    """完整的 Hybrid Automaton 规范 (Gemini 兼容)"""
    automaton: HAAutomatonSchema
    config: HAConfigSchema


# ==============================================================================
# Conversion Functions
# ==============================================================================

def _convert_reset_expr_to_reset_dict(reset_expr: str) -> Optional[Dict[str, list]]:
    """
    将 reset_expr 字符串转换为标准 reset 字典格式
    
    Examples:
        "x2=-0.9*x2[0]" -> {"x2": ["-0.9*x2[0]"]}
        "x1=0, x2=-0.9*x2[0]" -> {"x1": ["0"], "x2": ["-0.9*x2[0]"]}
        "" -> None
    """
    if not reset_expr or not reset_expr.strip():
        return None
    
    reset = {}
    # 支持逗号分隔的多个重置表达式
    for part in reset_expr.split(","):
        part = part.strip()
        if "=" in part:
            var, expr = part.split("=", 1)
            var = var.strip()
            expr = expr.strip()
            if var and expr:
                reset[var] = [expr]
    
    return reset if reset else None


def _convert_structured_to_standard_ha(structured_dict: dict) -> dict:
    """
    将结构化输出转换为标准 HA JSON 格式
    
    主要转换：reset_expr (string) -> reset (dict)
    """
    result = {
        "automaton": {
            "var": structured_dict["automaton"]["var"],
            "input": structured_dict["automaton"].get("input", ""),
            "mode": structured_dict["automaton"]["mode"],
            "edge": []
        },
        "config": {
            "dt": structured_dict["config"]["dt"],
            "total_time": structured_dict["config"]["total_time"],
        }
    }

    # 添加可选的 config 字段
    if structured_dict["config"].get("order"):
        result["config"]["order"] = structured_dict["config"]["order"]
    if structured_dict["config"].get("self_loop"):
        result["config"]["self_loop"] = True
    
    # 转换 edge (reset_expr -> reset)
    for edge in structured_dict["automaton"].get("edge", []):
        new_edge = {
            "direction": edge["direction"],
            "condition": edge["condition"]
        }
        # 转换 reset_expr 到 reset dict
        reset_expr = edge.get("reset_expr", "")
        reset_dict = _convert_reset_expr_to_reset_dict(reset_expr)
        if reset_dict:
            new_edge["reset"] = reset_dict
        
        result["automaton"]["edge"].append(new_edge)
    
    return result


def _get_structured_output_prompt() -> str:
    """获取结构化输出转换的 system prompt"""
    return """You are a format conversion expert. Convert the Hybrid Automaton description to precise JSON format.

CRITICAL FORMAT REQUIREMENTS:

1. **Equation format**: "var[order] = expression"
   - x[0] = variable value
   - x[1] = first derivative (dx/dt)
   - x[2] = second derivative
   - Example: "x1[1] = x2[0], x2[1] = -9.8"
   - Multiple equations in same mode: comma-separated

2. **Direction format**: "source -> target" with spaces
   - Example: "1 -> 2" (NOT "1->2")

3. **Condition format**: Use bare variable names
   - Example: "x1 <= 0" (NOT "x1[0] <= 0")

4. **Reset format**: String "var=expr" or "var1=expr1, var2=expr2"
   - Example: "x2=-0.9*x2[0]"
   - Use empty string "" if no reset

5. **Variables**: Keep exactly as described (don't add/remove)

Extract the HA specification and convert to the required JSON structure."""


def convert_agent_result_to_ha(
    agent_result: Any,
    model_id: str = "gemini/gemini-3-flash-preview",
    api_key: str = None,
    fallback_to_extraction: bool = True,
    timeout: int = 60,
    verbose: bool = True
) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """
    将 Agent 的输出转换为结构化的 HA JSON 规范（两阶段架构的第二阶段）
    
    流程:
    1. 先尝试传统提取方法 (extract_and_fix_ha_spec)
    2. 如果提取成功且验证通过，直接返回
    3. 如果提取失败或验证失败，使用 LLM 结构化输出转换
    
    Args:
        agent_result: Agent 的原始输出（字符串或字典）
        model_id: 用于结构化输出的 LLM 模型 ID
        api_key: API key（如果为 None，从环境变量获取）
        fallback_to_extraction: 如果结构化输出失败，是否回退到传统提取
        timeout: LLM 调用超时时间（秒）
        verbose: 是否打印详细信息
        
    Returns:
        Tuple of (ha_dict, success, message)
        - ha_dict: 转换后的 HA 规范字典（失败时为 None 或部分提取结果）
        - success: 是否成功
        - message: 状态消息
    """
    messages = []
    
    # ========== 阶段 1: 尝试传统提取 ==========
    if verbose:
        print("\n[StructuredOutput] Stage 1: Attempting traditional extraction...")
    
    ha_dict, extract_messages = extract_and_fix_ha_spec(agent_result, auto_fix=True)
    messages.extend(extract_messages)
    
    if ha_dict is not None:
        # 验证提取结果
        validation_result = validate_ha_with_schema(ha_dict)
        if validation_result['valid']:
            if verbose:
                print("[StructuredOutput] ✓ Traditional extraction successful and valid")
            return ha_dict, True, "Traditional extraction successful\n" + "\n".join(messages)
        else:
            if verbose:
                print(f"[StructuredOutput] Traditional extraction found but has {len(validation_result['errors'])} validation errors")
            messages.append(f"Extraction found but has {len(validation_result['errors'])} validation errors")
    else:
        if verbose:
            print("[StructuredOutput] Traditional extraction failed")
        messages.append("Traditional extraction failed")
    
    # ========== 阶段 2: 使用 LLM 结构化输出转换 ==========
    if not HAS_LITELLM:
        msg = "litellm not installed, cannot use structured output conversion"
        messages.append(msg)
        # 回退到传统提取结果（即使有验证错误）
        return ha_dict, ha_dict is not None, "\n".join(messages)
    
    if verbose:
        print("[StructuredOutput] Stage 2: Using LLM structured output conversion...")
    
    # 获取 API key
    if api_key is None:
        load_dotenv(override=True)
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        msg = "No GEMINI_API_KEY found, cannot use structured output"
        messages.append(msg)
        return ha_dict, ha_dict is not None, "\n".join(messages)
    
    # 准备输入文本
    if isinstance(agent_result, dict):
        input_text = json.dumps(agent_result, indent=2)
    else:
        input_text = str(agent_result)
    
    # 如果输入太长，截取关键部分
    if len(input_text) > 10000:
        # 尝试找到 JSON 块
        json_match = re.search(r'\{[\s\S]*"automaton"[\s\S]*\}', input_text)
        if json_match:
            input_text = json_match.group(0)
        else:
            # 取最后部分（通常是最终答案）
            input_text = input_text[-8000:]
    
    try:
        response = litellm.completion(
            model=model_id,
            messages=[
                {"role": "system", "content": _get_structured_output_prompt()},
                {"role": "user", "content": f"Convert this Hybrid Automaton description to JSON format:\n\n{input_text}"}
            ],
            api_key=api_key,
            timeout=timeout,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "HASpecification",
                    "schema": HASpecificationSchema.model_json_schema(),
                    "strict": True,
                }
            }
        )
        
        # 解析响应
        content = response.choices[0].message.content
        if content is None:
            msg = "LLM returned empty response (content is None)"
            messages.append(msg)
            if fallback_to_extraction and ha_dict is not None:
                messages.append("Falling back to traditional extraction result")
                return ha_dict, True, "\n".join(messages)
            return ha_dict, False, "\n".join(messages)
        structured_result = HASpecificationSchema.model_validate_json(content)

        # 转换为标准格式
        converted_ha = _convert_structured_to_standard_ha(structured_result.model_dump())
        
        # 验证转换结果
        validation_result = validate_ha_with_schema(converted_ha)
        if validation_result['valid']:
            if verbose:
                print("[StructuredOutput] ✓ LLM structured output conversion successful")
            messages.append("LLM structured output conversion successful")
            return converted_ha, True, "\n".join(messages)
        else:
            if verbose:
                print(f"[StructuredOutput] LLM output has {len(validation_result['errors'])} validation errors")
            messages.append(f"LLM output has validation errors: {validation_result['errors'][:3]}")
            
            # 返回转换结果（即使有小错误）
            return converted_ha, True, "\n".join(messages)
            
    except Exception as e:
        error_msg = f"LLM structured output failed: {type(e).__name__}: {str(e)[:200]}"
        if verbose:
            print(f"[StructuredOutput] ✗ {error_msg}")
        messages.append(error_msg)
    
    # ========== 回退处理 ==========
    if fallback_to_extraction and ha_dict is not None:
        messages.append("Falling back to traditional extraction result")
        return ha_dict, True, "\n".join(messages)
    
    return ha_dict, False, "\n".join(messages)


def preprocess_ha_for_evaluation_v2(
    agent_result: Any,
    model_id: str = "gemini/gemini-3-flash-preview",
    api_key: str = None,
    timeout: int = 60,
    verbose: bool = True
) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """
    使用 LLM 结构化输出将 Agent 结果转换为 HA JSON 规范
    
    Args:
        agent_result: Agent 的原始输出（字符串或字典）
        model_id: LLM 模型 ID
        api_key: API key（None 则从环境变量获取 GEMINI_API_KEY）
        timeout: LLM 调用超时时间（秒）
        verbose: 是否打印详细信息
        
    Returns:
        Tuple of (ha_dict, success, message)
    """
    # 检查 litellm 可用性
    if not HAS_LITELLM:
        return None, False, "Error: litellm not installed"
    
    # 获取 API key
    if api_key is None:
        load_dotenv(override=True)
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, False, "Error: No GEMINI_API_KEY found"
    
    # 准备输入文本
    input_text = json.dumps(agent_result, indent=2) if isinstance(agent_result, dict) else str(agent_result)
    
    # 输入过长时提取关键部分
    if len(input_text) > 10000:
        json_match = re.search(r'\{[\s\S]*"automaton"[\s\S]*\}', input_text)
        input_text = json_match.group(0) if json_match else input_text[-8000:]
    
    # 调用 LLM 进行结构化输出转换
    try:
        if verbose:
            print("[StructuredOutput] Converting with LLM...")
        
        response = litellm.completion(
            model=model_id,
            messages=[
                {"role": "system", "content": _get_structured_output_prompt()},
                {"role": "user", "content": f"Convert this Hybrid Automaton description to JSON format:\n\n{input_text}"}
            ],
            api_key=api_key,
            timeout=timeout,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "HASpecification",
                    "schema": HASpecificationSchema.model_json_schema(),
                    "strict": True,
                }
            }
        )
        
        # 解析并转换为标准格式
        content = response.choices[0].message.content
        if content is None:
            return None, False, "LLM returned empty response (content is None)"
        structured_result = HASpecificationSchema.model_validate_json(content)
        ha_dict = _convert_structured_to_standard_ha(structured_result.model_dump())
        
        # 验证结果
        validation = validate_ha_with_schema(ha_dict)
        if validation['valid']:
            if verbose:
                print("[StructuredOutput] ✓ Conversion successful")
            return ha_dict, True, "LLM structured output conversion successful"
        else:
            if verbose:
                print(f"[StructuredOutput] ✓ Converted with {len(validation['errors'])} validation warnings")
            return ha_dict, True, f"Converted with validation warnings: {validation['errors'][:3]}"
            
    except Exception as e:
        error_msg = f"LLM conversion failed: {type(e).__name__}: {str(e)[:200]}"
        if verbose:
            print(f"[StructuredOutput] ✗ {error_msg}")
        return None, False, error_msg


# ==============================================================================
# Standalone Test
# ==============================================================================

if __name__ == "__main__":
    # 测试结构化输出转换
    print("=" * 70)
    print("Testing HA Structured Output Converter")
    print("=" * 70)
    
    # 模拟 agent 输出
    test_agent_output = """
经过对轨迹数据的分析，我确定这是一个弹跳球系统。

系统分析结果：
- 状态变量: x1 (高度), x2 (速度)
- 无外部输入
- 单一模态：自由落体
- 动力学方程: dx1/dt = x2, dx2/dt = -9.8

边界条件：当球接触地面 (x1 <= 0) 时，速度反向并按系数 0.9 衰减。

最终 HA 规范:
```json
{
    "automaton": {
        "var": "x1, x2",
        "mode": [{"id": 1, "eq": "x1[1] = x2[0], x2[1] = -9.8"}],
        "edge": [{"direction": "1 -> 1", "condition": "x1 <= 0"}]
    },
    "config": {"dt": 0.001, "total_time": 10.0}
}
```

reset 应该设置为 x2 = -0.9 * x2[0]
"""
    
    print("\n输入:")
    print(test_agent_output[:500] + "...")
    
    # 测试转换
    ha_dict, success, message = convert_agent_result_to_ha(
        test_agent_output,
        model_id="gemini/gemini-2.0-flash",
        verbose=True
    )
    
    print(f"\n成功: {success}")
    print(f"消息: {message}")
    if ha_dict:
        print(f"\n转换结果:")
        print(json.dumps(ha_dict, indent=2))
