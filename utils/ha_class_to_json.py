"""
Conversion utilities for Python class-based Hybrid Automaton specifications to JSON format.

This module provides the conversion layer between the new Python class-based HA format
and the legacy JSON format used by HybridAutomata.from_json(). This allows gradual
migration without rewriting the existing simulator.
"""

from typing import Optional, Dict, Any
import numpy as np
import sys
import traceback


def convert_python_class_to_json(
    class_code: str,
    params: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Convert Python HA class to JSON format compatible with HybridAutomata.from_json().

    This function:
    1. Executes the class code in an isolated namespace
    2. Instantiates the HybridAutomaton class
    3. Optionally injects optimized parameters
    4. Calls the to_json() method to convert to JSON format

    Args:
        class_code: Python class definition as a string
        params: Optional numpy array of optimized parameters to inject.
                If None, uses the class's default params.

    Returns:
        Dictionary in JSON format compatible with HybridAutomata.from_json():
        {
            "automaton": {
                "var": "x1, x2, ...",
                "input": "u1, u2, ...",
                "mode": [...],
                "edge": [...]
            },
            "config": {
                "dt": 0.001,
                "total_time": 10.0,
                "order": 2,
                ...
            }
        }

    Raises:
        ValueError: If class code cannot be executed or converted
        AttributeError: If HybridAutomaton class is missing required methods
    """
    # Create isolated namespace for execution
    namespace = {}

    try:
        # Execute class code in isolated namespace
        exec(class_code, namespace)
    except Exception as e:
        raise ValueError(f"Failed to execute class code: {str(e)}\n{traceback.format_exc()}")

    # Check if HybridAutomaton class exists
    if 'HybridAutomaton' not in namespace:
        raise ValueError("Class code must define a class named 'HybridAutomaton'")

    try:
        # Instantiate the class
        ha_instance = namespace['HybridAutomaton']()
    except Exception as e:
        raise ValueError(f"Failed to instantiate HybridAutomaton class: {str(e)}\n{traceback.format_exc()}")

    # Inject optimized parameters if provided
    if params is not None:
        if isinstance(params, np.ndarray):
            ha_instance.params = params.tolist()
        elif isinstance(params, list):
            ha_instance.params = params
        else:
            raise ValueError(f"params must be np.ndarray or list, got {type(params)}")

    # Check if to_json method exists
    if not hasattr(ha_instance, 'to_json'):
        raise AttributeError("HybridAutomaton class must have a to_json() method")

    try:
        # Convert to JSON format
        ha_json = ha_instance.to_json()
    except Exception as e:
        raise ValueError(f"Failed to convert to JSON: {str(e)}\n{traceback.format_exc()}")

    # Validate JSON structure
    if not isinstance(ha_json, dict):
        raise ValueError(f"to_json() must return a dict, got {type(ha_json)}")

    if 'automaton' not in ha_json or 'config' not in ha_json:
        raise ValueError("JSON dict must contain 'automaton' and 'config' keys")

    return ha_json


def instantiate_class_and_get_params(class_code: str) -> tuple:
    """
    Instantiate the HA class and extract its default parameters.

    Args:
        class_code: Python class definition as a string

    Returns:
        Tuple of (ha_instance, params_list)
        where params_list is the default params from the class's __init__

    Raises:
        ValueError: If class cannot be instantiated
    """
    namespace = {}

    try:
        exec(class_code, namespace)
    except Exception as e:
        raise ValueError(f"Failed to execute class code: {str(e)}")

    if 'HybridAutomaton' not in namespace:
        raise ValueError("Class code must define 'HybridAutomaton'")

    try:
        ha_instance = namespace['HybridAutomaton']()
    except Exception as e:
        raise ValueError(f"Failed to instantiate class: {str(e)}")

    if not hasattr(ha_instance, 'params'):
        raise AttributeError("HybridAutomaton instance must have 'params' attribute")

    params = ha_instance.params
    if not isinstance(params, (list, np.ndarray)):
        raise ValueError(f"params must be list or np.ndarray, got {type(params)}")

    return ha_instance, params


def extract_metadata_from_class(class_code: str) -> Dict[str, Any]:
    """
    Extract metadata from HA class without full conversion.

    Useful for quick inspection of class properties.

    Args:
        class_code: Python class definition as a string

    Returns:
        Dictionary with metadata:
        {
            'num_variables': int,
            'num_inputs': int,
            'num_modes': int,
            'var': str,
            'input': str,
            'dt': float,
            'total_time': float,
            'order': int,
            'params': list
        }

    Raises:
        ValueError: If class cannot be instantiated
    """
    ha_instance, params = instantiate_class_and_get_params(class_code)

    # Extract variable names
    var_str = getattr(ha_instance, 'var', '')
    input_str = getattr(ha_instance, 'input', '')

    num_variables = len([v.strip() for v in var_str.split(',') if v.strip()])
    num_inputs = len([i.strip() for i in input_str.split(',') if i.strip()]) if input_str else 0

    # Get number of modes
    num_modes = 0
    if hasattr(ha_instance, 'num_modes'):
        try:
            num_modes = ha_instance.num_modes()
        except Exception:
            num_modes = 0

    metadata = {
        'num_variables': num_variables,
        'num_inputs': num_inputs,
        'num_modes': num_modes,
        'var': var_str,
        'input': input_str,
        'dt': getattr(ha_instance, 'dt', 0.001),
        'total_time': getattr(ha_instance, 'total_time', 10.0),
        'order': getattr(ha_instance, 'order', 1),
        'params': params if isinstance(params, list) else params.tolist()
    }

    return metadata


if __name__ == "__main__":
    # Example usage and testing
    example_class = '''
class HybridAutomaton:
    """Example HA class for testing."""

    def __init__(self):
        self.params = [-0.5, -5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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

    print("Testing ha_class_to_json.py")
    print("=" * 80)

    # Test 1: Convert with default params
    print("\nTest 1: Convert with default params")
    try:
        ha_json = convert_python_class_to_json(example_class)
        print("✓ Conversion successful")
        print(f"Variables: {ha_json['automaton']['var']}")
        print(f"Modes: {len(ha_json['automaton']['mode'])}")
        print(f"Config: dt={ha_json['config']['dt']}, total_time={ha_json['config']['total_time']}")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")

    # Test 2: Convert with custom params
    print("\nTest 2: Convert with optimized params")
    try:
        optimized_params = np.array([-0.3, -4.5, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ha_json = convert_python_class_to_json(example_class, optimized_params)
        print("✓ Conversion with optimized params successful")
        print(f"First mode equation: {ha_json['automaton']['mode'][0]['eq']}")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")

    # Test 3: Extract metadata
    print("\nTest 3: Extract metadata")
    try:
        metadata = extract_metadata_from_class(example_class)
        print("✓ Metadata extraction successful")
        print(f"Num variables: {metadata['num_variables']}")
        print(f"Num inputs: {metadata['num_inputs']}")
        print(f"Num modes: {metadata['num_modes']}")
        print(f"Params: {metadata['params']}")
    except Exception as e:
        print(f"✗ Metadata extraction failed: {e}")

    print("\n" + "=" * 80)
    print("All tests completed")
