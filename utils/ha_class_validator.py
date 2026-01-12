"""
Validation utilities for Python class-based Hybrid Automaton specifications.

This module handles extraction of Python classes from LLM output (markdown code blocks)
and validation of class syntax and structure.
"""

import re
import ast
from typing import Tuple, List, Optional
import inspect


def extract_python_class_from_text(text: str) -> Optional[str]:
    """
    Extract Python class code from text, handling markdown code blocks.

    Searches for:
    1. Code blocks with ```python or ``` markers
    2. Class definitions starting with 'class HybridAutomaton'

    Args:
        text: Input text containing Python code (may include markdown)

    Returns:
        Extracted Python class code as a string, or None if not found

    Examples:
        >>> text = '```python\\nclass HybridAutomaton:\\n    pass\\n```'
        >>> extract_python_class_from_text(text)
        'class HybridAutomaton:\\n    pass'
    """
    # Pattern 1: Look for code blocks with python marker
    pattern_python = r'```python\s*\n(.*?)```'
    matches_python = re.findall(pattern_python, text, re.DOTALL)

    # Pattern 2: Look for generic code blocks
    pattern_generic = r'```\s*\n(.*?)```'
    matches_generic = re.findall(pattern_generic, text, re.DOTALL)

    # Combine all matches
    all_matches = matches_python + matches_generic

    # Filter for blocks containing 'class HybridAutomaton'
    class_matches = [m for m in all_matches if 'class HybridAutomaton' in m]

    if class_matches:
        # Return the first match containing the class definition
        return class_matches[0].strip()

    # Fallback: Search for class definition directly in text (no markdown)
    pattern_direct = r'(class HybridAutomaton.*?)(?=\n(?:class\s|\Z))'
    match_direct = re.search(pattern_direct, text, re.DOTALL)

    if match_direct:
        return match_direct.group(1).strip()

    return None


def validate_python_class_syntax(class_code: str) -> Tuple[bool, List[str]]:
    """
    Validate Python class syntax and structure.

    Checks:
    1. Code compiles without syntax errors
    2. Class 'HybridAutomaton' is defined
    3. Required methods exist: __init__, num_modes, mode_dynamics, guard_condition, reset_map, to_json
    4. Method signatures are correct

    Args:
        class_code: Python class code as a string

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if all checks pass
        - error_messages: List of error messages (empty if valid)

    Examples:
        >>> code = "class HybridAutomaton:\\n    def __init__(self): pass"
        >>> is_valid, errors = validate_python_class_syntax(code)
        >>> is_valid
        False
        >>> 'num_modes' in ' '.join(errors)
        True
    """
    errors = []

    # Check 1: Syntax - Try to compile
    try:
        compile(class_code, '<string>', 'exec')
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return False, errors

    # Check 2: Parse AST to find class definition
    try:
        tree = ast.parse(class_code)
    except Exception as e:
        errors.append(f"Failed to parse code: {str(e)}")
        return False, errors

    # Find HybridAutomaton class
    ha_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'HybridAutomaton':
            ha_class = node
            break

    if ha_class is None:
        errors.append("No 'HybridAutomaton' class found in code")
        return False, errors

    # Check 3: Required methods
    required_methods = {
        '__init__': 1,      # 1 argument (self)
        'num_modes': 1,     # 1 argument (self)
        'mode_dynamics': 4, # 4 arguments (self, mode_id, x, u)
        'guard_condition': 5, # 5 arguments (self, source_mode, target_mode, x, u)
        'reset_map': 5,     # 5 arguments (self, source_mode, target_mode, x, u)
        'to_json': 1        # 1 argument (self)
    }

    # Extract method names from class
    class_methods = {}
    for item in ha_class.body:
        if isinstance(item, ast.FunctionDef):
            # Count arguments (including self)
            num_args = len(item.args.args)
            class_methods[item.name] = num_args

    # Check for missing methods
    for method_name, expected_args in required_methods.items():
        if method_name not in class_methods:
            errors.append(f"Missing required method: {method_name}()")
        elif class_methods[method_name] != expected_args:
            errors.append(
                f"Method {method_name}() has {class_methods[method_name]} arguments, "
                f"expected {expected_args}"
            )

    # Check 4: __init__ should define params attribute
    init_method = None
    for item in ha_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            init_method = item
            break

    if init_method:
        # Check if self.params is assigned
        has_params = False
        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == 'params':
                        has_params = True
                        break

        if not has_params:
            errors.append("__init__() method must define self.params attribute")

    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors


def auto_fix_common_class_issues(class_code: str) -> str:
    """
    Automatically fix common issues in Python class code.

    Fixes:
    1. self.params = (...) → self.params = [...] (tuple to list)
    2. Missing to_json() method → add stub
    3. Inconsistent indentation

    Args:
        class_code: Python class code as a string

    Returns:
        Fixed class code as a string
    """
    fixed_code = class_code

    # Fix 1: Convert params tuple to list
    # Pattern: self.params = (val1, val2, ...)
    pattern_tuple = r'(self\.params\s*=\s*)\((.*?)\)'
    if re.search(pattern_tuple, fixed_code, re.DOTALL):
        fixed_code = re.sub(pattern_tuple, r'\1[\2]', fixed_code, flags=re.DOTALL)

    # Fix 2: Check if to_json() method exists
    if 'def to_json(' not in fixed_code:
        # Add stub to_json() method at the end of the class
        # Find the last method and add after it
        lines = fixed_code.split('\n')

        # Find class indentation
        class_line_idx = -1
        for i, line in enumerate(lines):
            if 'class HybridAutomaton' in line:
                class_line_idx = i
                break

        if class_line_idx >= 0:
            # Determine method indentation (typically 4 spaces)
            method_indent = "    "

            # Add to_json stub
            stub = f'''
{method_indent}def to_json(self):
{method_indent}    """Convert to JSON format for compatibility with HybridAutomata.from_json()."""
{method_indent}    # TODO: Implement JSON conversion
{method_indent}    raise NotImplementedError("to_json() method not implemented")
'''
            # Append to end of class
            lines.append(stub)
            fixed_code = '\n'.join(lines)

    return fixed_code


def extract_initial_params_from_class(class_code: str) -> List[float]:
    """
    Extract the initial parameter values from the class's __init__ method.

    Args:
        class_code: Python class code as a string

    Returns:
        List of initial parameter values

    Raises:
        ValueError: If params cannot be extracted
    """
    # Execute class code in isolated namespace
    namespace = {}
    try:
        exec(class_code, namespace)
    except Exception as e:
        raise ValueError(f"Failed to execute class code: {str(e)}")

    # Instantiate and get params
    if 'HybridAutomaton' not in namespace:
        raise ValueError("No HybridAutomaton class found")

    try:
        ha_instance = namespace['HybridAutomaton']()
    except Exception as e:
        raise ValueError(f"Failed to instantiate class: {str(e)}")

    if not hasattr(ha_instance, 'params'):
        raise ValueError("Class instance has no 'params' attribute")

    params = ha_instance.params
    if isinstance(params, list):
        return params
    elif hasattr(params, 'tolist'):  # numpy array
        return params.tolist()
    else:
        raise ValueError(f"params must be list or array, got {type(params)}")


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
