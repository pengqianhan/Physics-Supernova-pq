"""Validation utilities for Python class-based Hybrid Automaton specifications.

Supports classes that inherit from HybridAutomatonBase.
"""

import re
import ast
from typing import Tuple, List, Optional
from dataclasses import dataclass

from utils.ha_base_class import HybridAutomatonBase


@dataclass
class StructuredAgentResult:
    """Structured result from agent containing analysis and code."""
    analysis_process: str
    class_code: str
    raw_output: str


def extract_python_class_from_text(text: str) -> Optional[str]:
    """Extract Python class code from text, handling markdown code blocks.

    Supports both standalone HybridAutomaton classes and classes inheriting from HybridAutomatonBase.
    """
    # Try markdown code blocks first
    for pattern in [r'```python\s*\n(.*?)```', r'```\s*\n(.*?)```']:
        for match in re.findall(pattern, text, re.DOTALL):
            # Check for either class definition pattern
            if ('class HybridAutomaton' in match or
                'HybridAutomatonBase' in match or
                re.search(r'class \w+\s*\(\s*HybridAutomatonBase\s*\)', match)):
                return match.strip()

    # Fallback: direct class definition
    # Match class HybridAutomaton or any class inheriting from HybridAutomatonBase
    patterns = [
        r'(class HybridAutomaton.*?)(?=\n*(?:final_answer|class\s|\Z))',
        r'(from utils\.ha_base_class.*?class \w+\s*\(\s*HybridAutomatonBase\s*\).*?)(?=\n*(?:final_answer|class\s|\Z))',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            lines = match.group(1).strip().split('\n')
            while lines and not lines[-1].strip():
                lines.pop()
            return '\n'.join(lines)

    return None


def extract_structured_result(text: str) -> StructuredAgentResult:
    """Extract analysis_process and class_code from agent output."""
    text_str = str(text)

    # Extract analysis section
    analysis = ""
    match = re.search(r'#{2,3}\s*Analysis\s*Process\s*\n(.*?)(?=#{2,3}\s*Python|```python|$)',
                      text_str, re.DOTALL | re.IGNORECASE)
    if match:
        analysis = match.group(1).strip()

    # Fallback: text before code block
    if not analysis:
        match = re.search(r'^(.*?)(?=```python|class\s+HybridAutomaton)', text_str, re.DOTALL)
        if match:
            potential = match.group(1).strip()
            if len(potential) > 50 and not potential.startswith('#'):
                analysis = potential

    return StructuredAgentResult(
        analysis_process=analysis,
        class_code=extract_python_class_from_text(text_str) or "",
        raw_output=text_str
    )


REQUIRED_METHODS = {
    '__init__': 1, 'num_modes': 1, 'mode_dynamics': 4,
    'guard_condition': 5, 'reset_map': 5
}


def validate_python_class_syntax(class_code: str) -> Tuple[bool, List[str]]:
    """Validate Python class syntax and structure.

    Supports both standalone HybridAutomaton classes and classes inheriting from HybridAutomatonBase.
    """
    errors = []

    # Syntax check
    try:
        tree = ast.parse(class_code)
    except SyntaxError as e:
        return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

    # Find class - either HybridAutomaton or any class inheriting from HybridAutomatonBase
    ha_class = None
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            # Check if it's named HybridAutomaton
            if n.name == 'HybridAutomaton':
                ha_class = n
                break
            # Check if it inherits from HybridAutomatonBase
            for base in n.bases:
                if isinstance(base, ast.Name) and base.id == 'HybridAutomatonBase':
                    ha_class = n
                    break
            if ha_class:
                break

    if not ha_class:
        return False, ["No 'HybridAutomaton' class or class inheriting from HybridAutomatonBase found"]

    # Check methods
    methods = {item.name: len(item.args.args)
               for item in ha_class.body if isinstance(item, ast.FunctionDef)}

    for name, expected in REQUIRED_METHODS.items():
        if name not in methods:
            errors.append(f"Missing: {name}()")
        elif methods[name] != expected:
            errors.append(f"{name}() has {methods[name]} args, expected {expected}")

    # Check params attribute
    init = next((item for item in ha_class.body
                 if isinstance(item, ast.FunctionDef) and item.name == '__init__'), None)
    if init:
        has_params = any(isinstance(n, ast.Assign) and
                        any(isinstance(t, ast.Attribute) and t.attr == 'params'
                            for t in n.targets)
                        for n in ast.walk(init))
        if not has_params:
            errors.append("__init__() must define self.params")

    return len(errors) == 0, errors


def auto_fix_common_class_issues(class_code: str) -> str:
    """Fix common issues: tuple to list for params, add to_json stub."""
    code = re.sub(r'(self\.params\s*=\s*)\((.*?)\)', r'\1[\2]', class_code, flags=re.DOTALL)

    if 'def to_json(' not in code:
        stub = '''
    def to_json(self):
        raise NotImplementedError("to_json() not implemented")
'''
        code = code.rstrip() + stub

    return code


def get_execution_namespace():
    """Get namespace with base class for executing HA class code."""
    class FakeModule:
        HybridAutomatonBase = HybridAutomatonBase

    return {
        'HybridAutomatonBase': HybridAutomatonBase,
        'utils': type('utils', (), {'ha_base_class': FakeModule})(),
    }


def extract_initial_params_from_class(class_code: str) -> List[float]:
    """Extract initial parameter values from class's __init__ method.

    Supports classes that inherit from HybridAutomatonBase.
    """
    # Create namespace with base class
    namespace = get_execution_namespace()

    # Filter out import statements (already in namespace)
    code_lines = class_code.split('\n')
    filtered_lines = []
    for line in code_lines:
        if 'from utils.ha_base_class import' in line:
            continue
        if 'import utils.ha_base_class' in line:
            continue
        filtered_lines.append(line)
    filtered_code = '\n'.join(filtered_lines)

    try:
        exec(filtered_code, namespace)
    except Exception as e:
        raise ValueError(f"Failed to execute: {e}")

    # Find the HA class
    ha_class = None
    if 'HybridAutomaton' in namespace:
        ha_class = namespace['HybridAutomaton']
    else:
        # Look for any class inheriting from HybridAutomatonBase
        for name, obj in namespace.items():
            if (isinstance(obj, type) and
                issubclass(obj, HybridAutomatonBase) and
                obj is not HybridAutomatonBase):
                ha_class = obj
                break

    if ha_class is None:
        raise ValueError("No HybridAutomaton class or class inheriting from HybridAutomatonBase found")

    try:
        ha = ha_class()
    except Exception as e:
        raise ValueError(f"Failed to instantiate: {e}")

    if not hasattr(ha, 'params'):
        raise ValueError("No 'params' attribute")

    params = ha.params
    return params.tolist() if hasattr(params, 'tolist') else list(params)


if __name__ == "__main__":
    # Test cases
    print("Testing ha_class_validator.py")
    print("=" * 80)

    # Test 1: Extract from markdown
    print("\nTest 1: Extract from markdown")
    markdown_text = '''
Here is the HA class:

```python
class HybridAutomaton:
    def __init__(self):
        self.params = [1.0, 2.0]

    def num_modes(self):
        return 1
```

That's the solution.
'''
    extracted = extract_python_class_from_text(markdown_text)
    if extracted and 'class HybridAutomaton' in extracted:
        print("OK: Successfully extracted class from markdown")
    else:
        print("FAIL: Failed to extract class")

    # Test 2: Extract class inheriting from base
    print("\nTest 2: Extract class inheriting from HybridAutomatonBase")
    markdown_text_inheritance = '''
Here is the HA class:

```python
from utils.ha_base_class import HybridAutomatonBase

class DuffingOscillator(HybridAutomatonBase):
    def __init__(self):
        self.params = [1.0, 2.0]

    def num_modes(self):
        return 1
```

That's the solution.
'''
    extracted = extract_python_class_from_text(markdown_text_inheritance)
    if extracted and 'HybridAutomatonBase' in extracted:
        print("OK: Successfully extracted class with inheritance")
    else:
        print("FAIL: Failed to extract class with inheritance")

    # Test 3: Validate valid class
    print("\nTest 3: Validate valid class")
    valid_class = '''
class HybridAutomaton:
    def __init__(self):
        self.params = [1.0, 2.0]
        self.var = "x1"

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        return "x1[1] = x1[0]"

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

    def to_json(self):
        return {"automaton": {}, "config": {}}
'''
    is_valid, errors = validate_python_class_syntax(valid_class)
    if is_valid:
        print("OK: Valid class passed validation")
    else:
        print(f"FAIL: Validation failed: {errors}")

    # Test 4: Validate class with inheritance
    print("\nTest 4: Validate class with inheritance")
    valid_class_inheritance = '''
class DuffingOscillator(HybridAutomatonBase):
    def __init__(self):
        self.params = [1.0, 2.0]
        self.var = "x1"

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        return "x1[1] = x1[0]"

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass
'''
    is_valid, errors = validate_python_class_syntax(valid_class_inheritance)
    if is_valid:
        print("OK: Valid class with inheritance passed validation")
    else:
        print(f"FAIL: Validation failed: {errors}")

    # Test 5: Extract params from class with inheritance
    print("\nTest 5: Extract params from class with inheritance")
    class_with_inheritance = '''
from utils.ha_base_class import HybridAutomatonBase

class DuffingOscillator(HybridAutomatonBase):
    def __init__(self):
        self.params = [1.5, 2.5, 3.5]
        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        return "x1[2] = x1[1] + x1[0]"

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

HybridAutomaton = DuffingOscillator
'''
    try:
        params = extract_initial_params_from_class(class_with_inheritance)
        if params == [1.5, 2.5, 3.5]:
            print(f"OK: Extracted params: {params}")
        else:
            print(f"FAIL: Wrong params: {params}")
    except Exception as e:
        print(f"FAIL: Failed to extract params: {e}")

    print("\n" + "=" * 80)
    print("All tests completed")
