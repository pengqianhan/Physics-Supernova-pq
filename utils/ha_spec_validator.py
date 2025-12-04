"""
Hybrid Automaton Specification Validator and Fixer

This module provides utilities to validate and fix HA specifications
before they are passed to HAEvaluator. Common LLM output errors are
detected and automatically corrected when possible.
"""

import re
import json
from typing import Dict, Any, Tuple, List, Optional


class HASpecValidationError(Exception):
    """Exception raised when HA specification validation fails."""
    def __init__(self, message: str, errors: List[str]):
        self.message = message
        self.errors = errors
        super().__init__(f"{message}: {'; '.join(errors)}")


def extract_dict_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a Python dictionary from text that may contain extra content.
    
    Handles common LLM output patterns:
    - Markdown code blocks (```python ... ```)
    - Extra text before/after the dict
    - JSON with Python boolean/None literals
    
    Args:
        text: Raw text containing a dictionary
        
    Returns:
        Extracted dictionary or None if extraction fails
    """
    if isinstance(text, dict):
        return text
    
    if not isinstance(text, str):
        return None
    
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Remove markdown code blocks
    code_block_pattern = r'```(?:python|json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    if matches:
        text = matches[-1]  # Take the last code block (likely the final answer)
    
    # Try to find dict boundaries
    # Look for {"automaton": ... pattern
    dict_start_pattern = r'\{\s*["\']automaton["\']'
    match = re.search(dict_start_pattern, text)
    if match:
        start_idx = match.start()
        # Find matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = start_idx + i + 1
                    break
        text = text[start_idx:end_idx]
    
    # Convert Python literals to JSON
    text = text.replace("True", "true").replace("False", "false").replace("None", "null")
    
    # Try JSON parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try Python eval (safer with ast.literal_eval)
    try:
        import ast
        # Convert JSON booleans back to Python for ast.literal_eval
        text_py = text.replace("true", "True").replace("false", "False").replace("null", "None")
        return ast.literal_eval(text_py)
    except (ValueError, SyntaxError):
        pass
    
    return None


def validate_ha_structure(ha_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the basic structure of an HA specification.
    
    Args:
        ha_dict: HA specification dictionary
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check top-level keys
    if not isinstance(ha_dict, dict):
        return False, ["HA specification must be a dictionary"]
    
    if 'automaton' not in ha_dict:
        errors.append("Missing required key 'automaton'")
    if 'config' not in ha_dict:
        errors.append("Missing required key 'config'")
    
    if errors:
        return False, errors
    
    automaton = ha_dict['automaton']
    config = ha_dict['config']
    
    # Validate automaton section
    if not isinstance(automaton, dict):
        errors.append("'automaton' must be a dictionary")
    else:
        # Check required automaton keys
        if 'var' not in automaton:
            errors.append("Missing 'automaton.var' (state variables)")
        elif not isinstance(automaton['var'], str) or not automaton['var'].strip():
            errors.append("'automaton.var' must be a non-empty string")
        
        if 'mode' not in automaton:
            errors.append("Missing 'automaton.mode' (mode list)")
        elif not isinstance(automaton['mode'], list):
            errors.append("'automaton.mode' must be a list")
        elif len(automaton['mode']) == 0:
            errors.append("'automaton.mode' must contain at least one mode")
        else:
            # Validate each mode
            for i, mode in enumerate(automaton['mode']):
                if not isinstance(mode, dict):
                    errors.append(f"Mode {i} must be a dictionary")
                    continue
                if 'id' not in mode:
                    errors.append(f"Mode {i} missing 'id'")
                elif not isinstance(mode['id'], (int, float)):
                    errors.append(f"Mode {i} 'id' must be an integer")
                if 'eq' not in mode:
                    errors.append(f"Mode {i} missing 'eq' (equations)")
                elif not isinstance(mode['eq'], str):
                    errors.append(f"Mode {i} 'eq' must be a string")
        
        if 'edge' not in automaton:
            errors.append("Missing 'automaton.edge' (edge list, can be empty [])")
        elif not isinstance(automaton['edge'], list):
            errors.append("'automaton.edge' must be a list")
        else:
            # Validate each edge
            for i, edge in enumerate(automaton['edge']):
                if not isinstance(edge, dict):
                    errors.append(f"Edge {i} must be a dictionary")
                    continue
                if 'direction' not in edge:
                    errors.append(f"Edge {i} missing 'direction'")
                elif not isinstance(edge['direction'], str):
                    errors.append(f"Edge {i} 'direction' must be a string")
                elif not re.match(r'\d+\s*->\s*\d+', edge['direction']):
                    errors.append(f"Edge {i} 'direction' must be 'source -> target' format (e.g., '1 -> 2')")
                if 'condition' not in edge:
                    errors.append(f"Edge {i} missing 'condition'")
                elif not isinstance(edge['condition'], str):
                    errors.append(f"Edge {i} 'condition' must be a string")
    
    # Validate config section
    if not isinstance(config, dict):
        errors.append("'config' must be a dictionary")
    else:
        if 'dt' not in config:
            errors.append("Missing 'config.dt' (time step)")
        elif not isinstance(config['dt'], (int, float)):
            errors.append("'config.dt' must be a number")
        
        if 'total_time' not in config:
            errors.append("Missing 'config.total_time'")
        elif not isinstance(config['total_time'], (int, float)):
            errors.append("'config.total_time' must be a number")
    
    return len(errors) == 0, errors


def validate_equation_syntax(eq_str: str, var_list: List[str], input_list: List[str] = None) -> Tuple[bool, str]:
    """
    Validate that an equation string can be parsed.
    
    Args:
        eq_str: Equation string (e.g., "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1")
        var_list: List of variable names
        input_list: List of input variable names
        
    Returns:
        Tuple of (is_valid, error_message or empty string)
    """
    if input_list is None:
        input_list = []
    
    try:
        # Split by comma for multiple equations
        equations = eq_str.split(',')
        for eq in equations:
            eq = eq.strip()
            if not eq:
                continue
            
            # Check for '=' sign
            if '=' not in eq:
                return False, f"Equation '{eq}' missing '=' sign"
            
            parts = eq.split('=')
            if len(parts) != 2:
                return False, f"Equation '{eq}' has invalid format (multiple '=' signs?)"
            
            lhs, rhs = parts[0].strip(), parts[1].strip()
            
            # Check LHS format: should be var_name[order]
            lhs_pattern = r'(\w+)\[(\d+)\]'
            lhs_match = re.match(lhs_pattern, lhs)
            if not lhs_match:
                return False, f"LHS '{lhs}' must be in format 'variable[order]' (e.g., x1[2])"
            
            var_name = lhs_match.group(1)
            if var_name not in var_list:
                return False, f"Variable '{var_name}' in LHS not found in var list: {var_list}"
            
            # Test if RHS can be converted to a lambda function
            # Replace var[n] with x[i][n] format
            pattern = r'(' + '|'.join(map(re.escape, var_list)) + r')\[(\d+)\]'
            
            def repl(match):
                string = match.group(1)
                number = match.group(2)
                idx = var_list.index(string)
                return f"x[{idx}][{number}]"
            
            test_expr = re.sub(pattern, repl, rhs)
            
            # Build lambda expression
            if input_list:
                input_expr = ',' + ','.join(input_list)
            else:
                input_expr = ""
            
            lambda_str = f"lambda x{input_expr}: {test_expr}"
            
            try:
                eval(lambda_str)
            except SyntaxError as e:
                return False, f"Equation '{eq}' has syntax error in RHS: {e}"
            except Exception as e:
                return False, f"Equation '{eq}' parsing error: {e}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Equation validation error: {e}"


def validate_condition_syntax(condition: str, var_list: List[str]) -> Tuple[bool, str]:
    """
    Validate that a guard condition string can be parsed.
    
    Args:
        condition: Condition string (e.g., "x1 > 0.5", "x2 <= 4 and x1 > 0")
        var_list: List of variable names
        
    Returns:
        Tuple of (is_valid, error_message or empty string)
    """
    try:
        # Convert x[0] format to x (if present)
        condition_clean = re.sub(r'(\w+)\[0\]', r'\1', condition)
        
        # Build lambda expression
        lambda_str = f"lambda {','.join(var_list)}: {condition_clean}"
        
        try:
            eval(lambda_str)
        except SyntaxError as e:
            return False, f"Condition '{condition}' has syntax error: {e}"
        except Exception as e:
            return False, f"Condition '{condition}' parsing error: {e}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Condition validation error: {e}"


def fix_common_issues(ha_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Attempt to fix common issues in HA specifications.
    
    Args:
        ha_dict: HA specification dictionary
        
    Returns:
        Tuple of (fixed_dict, list_of_fixes_applied)
    """
    fixes = []
    result = json.loads(json.dumps(ha_dict))  # Deep copy
    
    # Fix 1: Ensure mode IDs are integers
    if 'automaton' in result and 'mode' in result['automaton']:
        for mode in result['automaton']['mode']:
            if 'id' in mode and isinstance(mode['id'], float):
                mode['id'] = int(mode['id'])
                fixes.append(f"Converted mode id {mode['id']} from float to int")
    
    # Fix 2: Ensure edge list exists
    if 'automaton' in result:
        if 'edge' not in result['automaton']:
            result['automaton']['edge'] = []
            fixes.append("Added missing 'edge' key with empty list")
    
    # Fix 3: Fix direction format spacing
    if 'automaton' in result and 'edge' in result['automaton']:
        for edge in result['automaton']['edge']:
            if 'direction' in edge:
                # Normalize spacing: "1->2" or "1 ->2" -> "1 -> 2"
                direction = edge['direction']
                match = re.match(r'(\d+)\s*->\s*(\d+)', direction)
                if match:
                    new_direction = f"{match.group(1)} -> {match.group(2)}"
                    if new_direction != direction:
                        edge['direction'] = new_direction
                        fixes.append(f"Fixed direction format: '{direction}' -> '{new_direction}'")
    
    # Fix 4: Normalize condition syntax - remove x[0] if present (both formats are valid, but bare names preferred)
    if 'automaton' in result and 'edge' in result['automaton']:
        for edge in result['automaton']['edge']:
            if 'condition' in edge:
                original = edge['condition']
                # Remove [0] index from variable names in conditions (normalization, both formats work)
                normalized = re.sub(r'(\w+)\[0\]', r'\1', original)
                if normalized != original:
                    edge['condition'] = normalized
                    fixes.append(f"Normalized condition: '{original}' -> '{normalized}' (both formats valid)")
    
    # Fix 5: Ensure config has default values
    if 'config' in result:
        if 'dt' not in result['config']:
            result['config']['dt'] = 0.001
            fixes.append("Added default 'dt': 0.001")
        if 'total_time' not in result['config']:
            result['config']['total_time'] = 10.0
            fixes.append("Added default 'total_time': 10.0")
    
    # Fix 6: Ensure input field exists (can be empty string)
    if 'automaton' in result:
        if 'input' not in result['automaton']:
            result['automaton']['input'] = ""
            fixes.append("Added missing 'input' key with empty string")
    
    # Fix 7: Fix equations - ensure proper spacing
    if 'automaton' in result and 'mode' in result['automaton']:
        for mode in result['automaton']['mode']:
            if 'eq' in mode and isinstance(mode['eq'], str):
                eq = mode['eq']
                # Remove extra spaces around operators
                fixed_eq = re.sub(r'\s+', ' ', eq).strip()
                if fixed_eq != eq:
                    mode['eq'] = fixed_eq
                    fixes.append(f"Normalized whitespace in equation")
    
    return result, fixes


def validate_and_fix_ha_spec(ha_input: Any, auto_fix: bool = True) -> Tuple[Dict[str, Any], bool, List[str]]:
    """
    Main entry point: validate and optionally fix an HA specification.
    
    Args:
        ha_input: HA specification (can be dict or string)
        auto_fix: If True, attempt to fix common issues
        
    Returns:
        Tuple of (ha_dict, is_valid, messages)
        - ha_dict: The (possibly fixed) HA specification
        - is_valid: True if specification is valid (after fixes if applied)
        - messages: List of validation errors or fixes applied
    """
    messages = []
    
    # Step 1: Extract dict from input
    if isinstance(ha_input, str):
        ha_dict = extract_dict_from_text(ha_input)
        if ha_dict is None:
            return None, False, ["Failed to extract dictionary from text input"]
        messages.append("Extracted dictionary from text")
    elif isinstance(ha_input, dict):
        ha_dict = ha_input
    else:
        return None, False, [f"Invalid input type: {type(ha_input)}, expected dict or str"]
    
    # Step 2: Validate basic structure
    is_valid, errors = validate_ha_structure(ha_dict)
    if not is_valid:
        if not auto_fix:
            return ha_dict, False, errors
        messages.extend([f"Error: {e}" for e in errors])
    
    # Step 3: Apply fixes if enabled
    if auto_fix:
        ha_dict, fixes = fix_common_issues(ha_dict)
        messages.extend([f"Fixed: {f}" for f in fixes])
        
        # Re-validate after fixes
        is_valid, errors = validate_ha_structure(ha_dict)
        if not is_valid:
            messages.extend([f"Remaining error: {e}" for e in errors])
    
    # Step 4: Validate equation and condition syntax
    if is_valid and 'automaton' in ha_dict:
        automaton = ha_dict['automaton']
        var_list = [v.strip() for v in automaton.get('var', '').split(',') if v.strip()]
        input_list = [v.strip() for v in automaton.get('input', '').split(',') if v.strip()]
        
        # Validate each mode's equations
        for mode in automaton.get('mode', []):
            if 'eq' in mode:
                eq_valid, eq_error = validate_equation_syntax(mode['eq'], var_list, input_list)
                if not eq_valid:
                    is_valid = False
                    messages.append(f"Mode {mode.get('id', '?')} equation error: {eq_error}")
        
        # Validate each edge's condition
        for i, edge in enumerate(automaton.get('edge', [])):
            if 'condition' in edge:
                cond_valid, cond_error = validate_condition_syntax(edge['condition'], var_list)
                if not cond_valid:
                    is_valid = False
                    messages.append(f"Edge {i} condition error: {cond_error}")
    
    return ha_dict, is_valid, messages


# Convenience function for use in run_llm_ha_beta.py
def preprocess_ha_for_evaluation(agent_result: Any) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """
    Preprocess agent output for HAEvaluator.
    
    This is the main function to use before calling HAEvaluator.
    
    Args:
        agent_result: Raw result from agent.run()
        
    Returns:
        Tuple of (ha_spec, is_valid, status_message)
    """
    ha_dict, is_valid, messages = validate_and_fix_ha_spec(agent_result, auto_fix=True)
    
    status = "\n".join(messages) if messages else "Validation passed"
    
    if is_valid:
        return ha_dict, True, f"✓ HA specification validated successfully\n{status}"
    else:
        return ha_dict, False, f"✗ HA specification validation failed\n{status}"


if __name__ == "__main__":
    # Test cases
    print("=" * 60)
    print("Testing HA Specification Validator")
    print("=" * 60)
    
    # Test 1: Valid specification
    valid_spec = {
        "automaton": {
            "var": "x1",
            "input": "u1",
            "mode": [{"id": 1, "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1"}],
            "edge": []
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    
    result, is_valid, messages = validate_and_fix_ha_spec(valid_spec)
    print(f"\nTest 1 (Valid spec): {'PASS' if is_valid else 'FAIL'}")
    print(f"Messages: {messages}")
    
    # Test 2: Spec with issues that can be fixed
    fixable_spec = {
        "automaton": {
            "var": "x1, x2",
            "input": "u1",
            "mode": [{"id": 1.0, "eq": "x1[1] = x2[0], x2[1] = -9.8 + u1"}],  # float id
            "edge": [
                {"direction": "1->1", "condition": "x1[0] <= 0", "reset": {"x2": ["-0.9 * x2[0]"]}}  # wrong format
            ]
        },
        "config": {"dt": 0.001}  # missing total_time
    }
    
    result, is_valid, messages = validate_and_fix_ha_spec(fixable_spec)
    print(f"\nTest 2 (Fixable spec): {'PASS' if is_valid else 'FAIL'}")
    print(f"Messages: {messages}")
    print(f"Fixed result: {json.dumps(result, indent=2)}")
    
    # Test 3: Text extraction
    text_input = '''Here is the HA specification:
```python
{"automaton": {"var": "x1", "mode": [{"id": 1, "eq": "x1[1] = -2 * x1[0]"}], "edge": []}, "config": {"dt": 0.001, "total_time": 10.0}}
```
    '''
    
    result, is_valid, messages = validate_and_fix_ha_spec(text_input)
    print(f"\nTest 3 (Text extraction): {'PASS' if is_valid else 'FAIL'}")
    print(f"Messages: {messages}")

