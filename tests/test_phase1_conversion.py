"""
Phase 1 Integration Test: Python Class → JSON → Simulator Compatibility

This test verifies that the conversion infrastructure works end-to-end:
1. Hand-written Python class
2. Validation
3. Conversion to JSON
4. Compatibility with HybridAutomata.from_json()
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ha_class_validator import (
    extract_python_class_from_text,
    validate_python_class_syntax,
    extract_initial_params_from_class
)
from utils.ha_class_to_json import (
    convert_python_class_to_json,
    extract_metadata_from_class
)

# Add Dainarx code to path for simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils', 'Dainarx_code'))
from src.HybridAutomata import HybridAutomata


def test_single_mode_oscillator():
    """Test 1: Single-mode damped oscillator."""
    print("\n" + "=" * 80)
    print("Test 1: Single-Mode Damped Harmonic Oscillator")
    print("=" * 80)

    class_code = '''
class HybridAutomaton:
    """Damped harmonic oscillator."""

    def __init__(self):
        self.params = [
            -0.1,  # params[0]: damping
            -5.0,  # params[1]: spring constant
            1.0,   # params[2]: input gain
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2
        self.need_reset = False
        self.non_linear_items = ""

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u['u1']"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

    def to_json(self):
        mode_eq = f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u1"
        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [{"id": 1, "eq": mode_eq}],
                "edge": []
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
'''

    # Step 1: Validate syntax
    print("\n[Step 1] Validating Python class syntax...")
    is_valid, errors = validate_python_class_syntax(class_code)
    if is_valid:
        print("✓ Syntax validation PASSED")
    else:
        print(f"✗ Syntax validation FAILED: {errors}")
        return False

    # Step 2: Extract params
    print("\n[Step 2] Extracting initial params...")
    try:
        params = extract_initial_params_from_class(class_code)
        print(f"✓ Params extracted: {params[:3]}")
    except Exception as e:
        print(f"✗ Param extraction FAILED: {e}")
        return False

    # Step 3: Convert to JSON
    print("\n[Step 3] Converting to JSON format...")
    try:
        ha_json = convert_python_class_to_json(class_code)
        print(f"✓ JSON conversion PASSED")
        print(f"  Variables: {ha_json['automaton']['var']}")
        print(f"  Modes: {len(ha_json['automaton']['mode'])}")
        print(f"  Edges: {len(ha_json['automaton']['edge'])}")
    except Exception as e:
        print(f"✗ JSON conversion FAILED: {e}")
        return False

    # Step 4: Verify simulator compatibility
    print("\n[Step 4] Testing HybridAutomata.from_json() compatibility...")
    try:
        ha_system = HybridAutomata.from_json(ha_json["automaton"])
        print(f"✓ Simulator compatibility PASSED")
        print(f"  Created HA with {len(ha_system.mode_list)} mode(s)")
    except Exception as e:
        print(f"✗ Simulator compatibility FAILED: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ Test 1 PASSED: Single-mode oscillator")
    print("=" * 80)
    return True


def test_two_mode_with_switching():
    """Test 2: Two-mode system with threshold switching."""
    print("\n" + "=" * 80)
    print("Test 2: Two-Mode System with Threshold Switching")
    print("=" * 80)

    class_code = '''
class HybridAutomaton:
    """Two-mode oscillator with threshold switching."""

    def __init__(self):
        self.params = [
            -0.1,  # params[0]: damping mode 1
            -5.0,  # params[1]: spring constant
            -0.5,  # params[2]: damping mode 2
            0.0,   # params[3]: threshold
            1.0,   # params[4]: input gain
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2
        self.need_reset = False
        self.non_linear_items = ""

    def num_modes(self):
        return 2

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u['u1']"
        elif mode_id == 2:
            return f"x1[2] = {self.params[2]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u['u1']"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        if source_mode == 1 and target_mode == 2:
            return x['x1'][0] >= self.params[3]
        elif source_mode == 2 and target_mode == 1:
            return x['x1'][0] < self.params[3]
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

    def to_json(self):
        mode_1_eq = f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u1"
        mode_2_eq = f"x1[2] = {self.params[2]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u1"
        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [
                    {"id": 1, "eq": mode_1_eq},
                    {"id": 2, "eq": mode_2_eq}
                ],
                "edge": [
                    {"direction": "1 -> 2", "condition": f"x1 >= {self.params[3]}"},
                    {"direction": "2 -> 1", "condition": f"x1 < {self.params[3]}"}
                ]
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
'''

    # Validate
    print("\n[Step 1] Validating...")
    is_valid, errors = validate_python_class_syntax(class_code)
    if not is_valid:
        print(f"✗ Validation FAILED: {errors}")
        return False
    print("✓ Validation PASSED")

    # Convert
    print("\n[Step 2] Converting to JSON...")
    try:
        ha_json = convert_python_class_to_json(class_code)
        print(f"✓ Conversion PASSED")
        print(f"  Modes: {len(ha_json['automaton']['mode'])}")
        print(f"  Edges: {len(ha_json['automaton']['edge'])}")
    except Exception as e:
        print(f"✗ Conversion FAILED: {e}")
        return False

    # Test simulator
    print("\n[Step 3] Testing simulator compatibility...")
    try:
        ha_system = HybridAutomata.from_json(ha_json["automaton"])
        print(f"✓ Simulator PASSED")
        print(f"  Modes: {len(ha_system.mode_list)}")
        print(f"  Transitions: {len(ha_system.adj)}")
    except Exception as e:
        print(f"✗ Simulator FAILED: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ Test 2 PASSED: Two-mode system")
    print("=" * 80)
    return True


def test_markdown_extraction():
    """Test 3: Extract class from markdown code blocks."""
    print("\n" + "=" * 80)
    print("Test 3: Markdown Code Block Extraction")
    print("=" * 80)

    markdown_text = '''
Here is my solution:

```python
class HybridAutomaton:
    def __init__(self):
        self.params = [-0.1, -5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.var = "x1"
        self.input = ""
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2
        self.need_reset = False
        self.non_linear_items = ""

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0]"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

    def to_json(self):
        mode_eq = f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0]"
        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [{"id": 1, "eq": mode_eq}],
                "edge": []
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
```

This should work!
'''

    print("\n[Step 1] Extracting from markdown...")
    extracted = extract_python_class_from_text(markdown_text)
    if extracted and 'class HybridAutomaton' in extracted:
        print("✓ Extraction PASSED")
    else:
        print("✗ Extraction FAILED")
        return False

    print("\n[Step 2] Validating extracted class...")
    is_valid, errors = validate_python_class_syntax(extracted)
    if is_valid:
        print("✓ Validation PASSED")
    else:
        print(f"✗ Validation FAILED: {errors}")
        return False

    print("\n[Step 3] Converting to JSON...")
    try:
        ha_json = convert_python_class_to_json(extracted)
        print("✓ Conversion PASSED")
    except Exception as e:
        print(f"✗ Conversion FAILED: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ Test 3 PASSED: Markdown extraction")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PHASE 1 INTEGRATION TEST" + " " * 34 + "║")
    print("║" + " " * 15 + "Python Class → JSON → Simulator" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")

    results = []

    # Run tests
    results.append(("Single-mode oscillator", test_single_mode_oscillator()))
    results.append(("Two-mode with switching", test_two_mode_with_switching()))
    results.append(("Markdown extraction", test_markdown_extraction()))

    # Summary
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "TEST SUMMARY" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")

    total = len(results)
    passed = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 Phase 1 COMPLETED: All conversion infrastructure tests passed!")
        print("Ready to proceed to Phase 2: Parameter Optimization\n")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review and fix before proceeding.\n")
        sys.exit(1)
