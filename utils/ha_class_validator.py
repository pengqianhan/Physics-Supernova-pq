"""Validation utilities for Python class-based Hybrid Automaton specifications."""

import re
import ast
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class StructuredAgentResult:
    """Structured result from agent containing analysis and code."""
    analysis_process: str
    class_code: str
    raw_output: str


def extract_python_class_from_text(text: str) -> Optional[str]:
    """Extract Python class code from text, handling markdown code blocks."""
    # Try markdown code blocks first
    for pattern in [r'```python\s*\n(.*?)```', r'```\s*\n(.*?)```']:
        for match in re.findall(pattern, text, re.DOTALL):
            if 'class HybridAutomaton' in match:
                return match.strip()

    # Fallback: direct class definition
    match = re.search(r'(class HybridAutomaton.*?)(?=\n*(?:final_answer|class\s|\Z))', text, re.DOTALL)
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
    """Validate Python class syntax and structure."""
    errors = []

    # Syntax check
    try:
        tree = ast.parse(class_code)
    except SyntaxError as e:
        return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

    # Find class
    ha_class = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == 'HybridAutomaton'), None)
    if not ha_class:
        return False, ["No 'HybridAutomaton' class found"]

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
    """Fix common issues: tuple→list for params, add to_json stub."""
    code = re.sub(r'(self\.params\s*=\s*)\((.*?)\)', r'\1[\2]', class_code, flags=re.DOTALL)

    if 'def to_json(' not in code:
        stub = '''
    def to_json(self):
        raise NotImplementedError("to_json() not implemented")
'''
        code = code.rstrip() + stub

    return code


def extract_initial_params_from_class(class_code: str) -> List[float]:
    """Extract initial parameter values from class's __init__ method."""
    namespace = {}
    try:
        exec(class_code, namespace)
    except Exception as e:
        raise ValueError(f"Failed to execute: {e}")

    if 'HybridAutomaton' not in namespace:
        raise ValueError("No HybridAutomaton class found")

    try:
        ha = namespace['HybridAutomaton']()
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
        print("✓ Successfully extracted class from markdown")
    else:
        print("✗ Failed to extract class")

    # Test 2: Validate valid class
    print("\nTest 2: Validate valid class")
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
        print("✓ Valid class passed validation")
    else:
        print(f"✗ Validation failed: {errors}")

    # Test 3: Validate invalid class (missing methods)
    print("\nTest 3: Validate invalid class (missing methods)")
    invalid_class = '''
class HybridAutomaton:
    def __init__(self):
        self.params = [1.0, 2.0]
'''
    is_valid, errors = validate_python_class_syntax(invalid_class)
    if not is_valid and len(errors) > 0:
        print(f"✓ Invalid class correctly identified ({len(errors)} errors)")
        print(f"  Errors: {errors[:2]}")  # Show first 2 errors
    else:
        print("✗ Should have failed validation")

    # Test 4: Auto-fix params tuple
    print("\nTest 4: Auto-fix params tuple")
    class_with_tuple = '''
class HybridAutomaton:
    def __init__(self):
        self.params = (1.0, 2.0, 3.0)
'''
    fixed = auto_fix_common_class_issues(class_with_tuple)
    if 'self.params = [1.0, 2.0, 3.0]' in fixed:
        print("✓ Successfully fixed tuple to list")
    else:
        print("✗ Failed to fix tuple")

    # Test 5: Extract params
    print("\nTest 5: Extract initial params")
    try:
        params = extract_initial_params_from_class(valid_class)
        print(f"✓ Extracted params: {params}")
    except Exception as e:
        print(f"✗ Failed to extract params: {e}")

    print("\n" + "=" * 80)
    print("All tests completed")
