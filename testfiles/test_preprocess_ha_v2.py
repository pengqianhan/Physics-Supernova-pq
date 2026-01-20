"""
Test file for preprocess_ha_for_evaluation_v2

Usage:
    python test_preprocess_ha_v2.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ha_structured_output import (
    preprocess_ha_for_evaluation_v2,
    convert_agent_result_to_ha,
    _convert_reset_expr_to_reset_dict,
    _convert_structured_to_standard_ha,
)


def test_convert_reset_expr_to_reset_dict():
    """Test reset expression conversion"""
    print("\n" + "=" * 60)
    print("Test: _convert_reset_expr_to_reset_dict")
    print("=" * 60)
    
    # Test case 1: Simple reset
    result = _convert_reset_expr_to_reset_dict("x2=-0.9*x2[0]")
    expected = {"x2": ["-0.9*x2[0]"]}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ Simple reset: {result}")
    
    # Test case 2: Multiple resets
    result = _convert_reset_expr_to_reset_dict("x1=0, x2=-0.9*x2[0]")
    expected = {"x1": ["0"], "x2": ["-0.9*x2[0]"]}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ Multiple resets: {result}")
    
    # Test case 3: Empty string
    result = _convert_reset_expr_to_reset_dict("")
    assert result is None, f"Expected None, got {result}"
    print(f"✓ Empty string: {result}")
    
    # Test case 4: Whitespace only
    result = _convert_reset_expr_to_reset_dict("   ")
    assert result is None, f"Expected None, got {result}"
    print(f"✓ Whitespace only: {result}")
    
    print("\nAll reset conversion tests passed!")


def test_convert_structured_to_standard_ha():
    """Test structured to standard HA conversion"""
    print("\n" + "=" * 60)
    print("Test: _convert_structured_to_standard_ha")
    print("=" * 60)
    
    # Test case: Full conversion with edges and reset
    structured_dict = {
        "automaton": {
            "var": "x1, x2",
            "input": "",
            "mode": [
                {"id": 1, "eq": "x1[1] = x2[0], x2[1] = -9.8"}
            ],
            "edge": [
                {
                    "direction": "1 -> 1",
                    "condition": "x1 <= 0",
                    "reset_expr": "x2=-0.9*x2[0]"
                }
            ]
        },
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "self_loop": True
        }
    }
    
    result = _convert_structured_to_standard_ha(structured_dict)
    
    # Check basic structure
    assert "automaton" in result, "Missing 'automaton' key"
    assert "config" in result, "Missing 'config' key"
    print(f"✓ Basic structure present")
    
    # Check var and input
    assert result["automaton"]["var"] == "x1, x2", "Incorrect var"
    assert result["automaton"]["input"] == "", "Incorrect input"
    print(f"✓ var and input correct")
    
    # Check mode
    assert len(result["automaton"]["mode"]) == 1, "Incorrect mode count"
    assert result["automaton"]["mode"][0]["id"] == 1, "Incorrect mode id"
    print(f"✓ Mode correct")
    
    # Check edge with reset conversion
    assert len(result["automaton"]["edge"]) == 1, "Incorrect edge count"
    edge = result["automaton"]["edge"][0]
    assert edge["direction"] == "1 -> 1", "Incorrect edge direction"
    assert edge["condition"] == "x1 <= 0", "Incorrect edge condition"
    assert "reset" in edge, "Missing reset in edge"
    assert edge["reset"] == {"x2": ["-0.9*x2[0]"]}, f"Incorrect reset conversion: {edge['reset']}"
    print(f"✓ Edge with reset conversion correct")
    
    # Check config
    assert result["config"]["dt"] == 0.001, "Incorrect dt"
    assert result["config"]["total_time"] == 10.0, "Incorrect total_time"
    assert result["config"]["self_loop"] == True, "Missing self_loop"
    print(f"✓ Config correct")
    
    print(f"\nConverted result:\n{json.dumps(result, indent=2)}")
    print("\nAll structured conversion tests passed!")


def test_preprocess_ha_for_evaluation_v2_with_valid_json():
    """Test preprocess with already valid JSON input"""
    print("\n" + "=" * 60)
    print("Test: preprocess_ha_for_evaluation_v2 (with valid JSON)")
    print("=" * 60)
    
    # Valid HA specification as input
    valid_ha_json = """
{
    "automaton": {
        "var": "x1, x2",
        "input": "",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x2[0], x2[1] = -9.8"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 1",
                "condition": "x1 <= 0",
                "reset": {"x2": ["-0.9*x2[0]"]}
            }
        ]
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0
    }
}
"""
    
    print(f"Input: Valid JSON HA spec")
    
    # This will likely succeed with traditional extraction
    ha_dict, success, message = convert_agent_result_to_ha(
        valid_ha_json,
        verbose=True
    )
    
    print(f"\nSuccess: {success}")
    print(f"Message: {message}")
    if ha_dict:
        print(f"\nExtracted HA spec:")
        print(json.dumps(ha_dict, indent=2))
    
    return success


def test_preprocess_ha_for_evaluation_v2_with_text():
    """Test preprocess with mixed text/JSON input (requires LLM)"""
    print("\n" + "=" * 60)
    print("Test: preprocess_ha_for_evaluation_v2 (with text + JSON)")
    print("=" * 60)
    
    # Simulated agent output with text description
    agent_output = """
After analyzing the trajectory data, I have identified this as a bouncing ball system.

System Analysis:
- State variables: x1 (height), x2 (velocity)
- No external inputs
- Single mode: free fall dynamics
- Dynamics: dx1/dt = x2, dx2/dt = -9.8

Boundary condition: When ball hits ground (x1 <= 0), velocity reverses with coefficient 0.9.

Final HA Specification:
```json
{
    "automaton": {
        "var": "x1, x2",
        "input": "",
        "mode": [{"id": 1, "eq": "x1[1] = x2[0], x2[1] = -9.8"}],
        "edge": [{"direction": "1 -> 1", "condition": "x1 <= 0"}]
    },
    "config": {"dt": 0.001, "total_time": 10.0}
}
```

The reset should be x2 = -0.9 * x2[0]
"""
    
    print(f"Input: Mixed text + JSON")
    
    # First try convert_agent_result_to_ha (two-stage)
    ha_dict, success, message = convert_agent_result_to_ha(
        agent_output,
        model_id="gemini/gemini-3-flash-preview",
        verbose=True
    )
    
    print(f"\nSuccess: {success}")
    print(f"Message: {message}")
    if ha_dict:
        print(f"\nExtracted HA spec:")
        print(json.dumps(ha_dict, indent=2))
    
    return success


def test_preprocess_ha_for_evaluation_v2_direct():
    """Test preprocess_ha_for_evaluation_v2 directly (always uses LLM)"""
    print("\n" + "=" * 60)
    print("Test: preprocess_ha_for_evaluation_v2 (direct LLM call)")
    print("=" * 60)
    
    # More complex agent output
    agent_output = """
经过分析，这是一个 Duffing 振子系统。

分析结果:
- 状态变量: x1 (位移)
- 输入: u1 (外力)
- 方程: d²x/dt² = -0.1*dx/dt - x - x³ + u1

HA 规范:
```json
{
    "automaton": {
        "var": "x1",
        "input": "u1",
        "mode": [{"id": 1, "eq": "x1[2] = -0.1*x1[1] - x1[0] - x1[0]**3 + u1"}],
        "edge": []
    },
    "config": {"dt": 0.001, "total_time": 10.0}
}
```
"""
    
    print(f"Input: Chinese text + JSON (Duffing oscillator)")
    
    ha_dict, success, message = preprocess_ha_for_evaluation_v2(
        agent_output,
        model_id="gemini/gemini-3-flash-preview",
        verbose=True
    )
    
    print(f"\nSuccess: {success}")
    print(f"Message: {message}")
    if ha_dict:
        print(f"\nExtracted HA spec:")
        print(json.dumps(ha_dict, indent=2))
    
    return success


def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing preprocess_ha_for_evaluation_v2 and related functions")
    print("=" * 70)
    
    # Unit tests (no API calls)
    test_convert_reset_expr_to_reset_dict()
    test_convert_structured_to_standard_ha()
    
    # Integration tests (may require API key)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n" + "!" * 60)
        print("WARNING: GEMINI_API_KEY not set")
        print("Skipping LLM-based tests. Set GEMINI_API_KEY to run full tests.")
        print("!" * 60)
        return
    
    print("\n" + "-" * 60)
    print("Running integration tests (requires GEMINI_API_KEY)")
    print("-" * 60)
    
    # Run integration tests
    test_preprocess_ha_for_evaluation_v2_with_valid_json()
    test_preprocess_ha_for_evaluation_v2_with_text()
    test_preprocess_ha_for_evaluation_v2_direct()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
