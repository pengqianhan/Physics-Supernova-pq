"""
Hybrid Automaton Specification Extractor and Fixer

This module provides utilities to extract and fix HA specifications
before they are passed to JSON Schema validation. Common LLM output errors are
detected and automatically corrected when possible.

Note: Validation logic has been moved to ha_json_schema.py to avoid duplication.
This module focuses on:
1. Extracting dictionaries from LLM text output
2. Auto-fixing common issues (float IDs, direction format, etc.)
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


def extract_and_fix_ha_spec(ha_input: Any, auto_fix: bool = True) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Extract and optionally fix an HA specification from input.
    
    Note: Validation is handled by ha_json_schema.validate_ha_with_schema()
    
    Args:
        ha_input: HA specification (can be dict or string)
        auto_fix: If True, attempt to fix common issues
        
    Returns:
        Tuple of (ha_dict, messages)
        - ha_dict: The (possibly fixed) HA specification, or None if extraction failed
        - messages: List of fixes applied or extraction errors
    """
    messages = []
    
    # Step 1: Extract dict from input
    if isinstance(ha_input, str):
        ha_dict = extract_dict_from_text(ha_input)
        if ha_dict is None:
            return None, ["Failed to extract dictionary from text input"]
        messages.append("Extracted dictionary from text")
    elif isinstance(ha_input, dict):
        ha_dict = json.loads(json.dumps(ha_input))  # Deep copy
    else:
        return None, [f"Invalid input type: {type(ha_input)}, expected dict or str"]
    
    # Step 2: Apply fixes if enabled
    if auto_fix:
        ha_dict, fixes = fix_common_issues(ha_dict)
        messages.extend([f"Fixed: {f}" for f in fixes])
    
    return ha_dict, messages


# Keep old function name for backward compatibility
def validate_and_fix_ha_spec(ha_input: Any, auto_fix: bool = True) -> Tuple[Optional[Dict[str, Any]], bool, List[str]]:
    """
    Legacy function for backward compatibility.
    Extracts and fixes HA spec, returns a compatibility tuple.
    
    Note: Actual validation should be done via ha_json_schema.validate_ha_with_schema()
    
    Args:
        ha_input: HA specification (can be dict or string)
        auto_fix: If True, attempt to fix common issues
        
    Returns:
        Tuple of (ha_dict, is_extracted, messages)
        - ha_dict: The (possibly fixed) HA specification
        - is_extracted: True if extraction succeeded (NOT validation result)
        - messages: List of fixes applied
    """
    ha_dict, messages = extract_and_fix_ha_spec(ha_input, auto_fix)
    is_extracted = ha_dict is not None
    return ha_dict, is_extracted, messages


# Convenience function for use in run_llm_ha_beta.py
def preprocess_ha_for_evaluation(agent_result: Any) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """
    Preprocess agent output for HAEvaluator.
    
    This function extracts and fixes the HA spec. For full validation,
    use ha_json_schema.validate_ha_with_schema() on the returned dict.
    
    Args:
        agent_result: Raw result from agent.run()
        
    Returns:
        Tuple of (ha_spec, is_extracted, status_message)
    """
    ha_dict, messages = extract_and_fix_ha_spec(agent_result, auto_fix=True)
    
    status = "\n".join(messages) if messages else "Extraction successful"
    
    if ha_dict is not None:
        return ha_dict, True, f"✓ HA specification extracted and fixed\n{status}"
    else:
        return ha_dict, False, f"✗ HA specification extraction failed\n{status}"


if __name__ == "__main__":
    # Test cases
    print("=" * 60)
    print("Testing HA Specification Extractor and Fixer")
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
    
    result, messages = extract_and_fix_ha_spec(valid_spec)
    print(f"\nTest 1 (Valid spec): {'PASS' if result else 'FAIL'}")
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
    
    result, messages = extract_and_fix_ha_spec(fixable_spec)
    print(f"\nTest 2 (Fixable spec): {'PASS' if result else 'FAIL'}")
    print(f"Messages: {messages}")
    print(f"Fixed result: {json.dumps(result, indent=2)}")
    
    # Test 3: Text extraction
    text_input = '''Here is the HA specification:
```python
{"automaton": {"var": "x1", "mode": [{"id": 1, "eq": "x1[1] = -2 * x1[0]"}], "edge": []}, "config": {"dt": 0.001, "total_time": 10.0}}
```
    '''
    
    result, messages = extract_and_fix_ha_spec(text_input)
    print(f"\nTest 3 (Text extraction): {'PASS' if result else 'FAIL'}")
    print(f"Messages: {messages}")

