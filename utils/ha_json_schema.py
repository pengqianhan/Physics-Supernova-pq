"""
JSON Schema Definition and Validation for Hybrid Automaton Specifications

This module provides:
1. A comprehensive JSON Schema for HA specifications
2. Schema-based validation using jsonschema library
3. Semantic validation beyond schema (equation syntax, condition syntax)
4. Clear error messages with path information

Usage:
    from utils.ha_json_schema import validate_ha_with_schema, HA_JSON_SCHEMA

    # Validate an HA specification
    result = validate_ha_with_schema(ha_dict)
    if result['valid']:
        print("Specification is valid")
    else:
        for error in result['errors']:
            print(f"Error at {error['path']}: {error['message']}")
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from jsonschema import validate, ValidationError, Draft7Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("Warning: jsonschema not installed. Install with: pip install jsonschema")


# ==============================================================================
# JSON Schema Definition for Hybrid Automaton Specifications
# ==============================================================================

HA_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://example.com/hybrid-automaton.schema.json",
    "title": "Hybrid Automaton Specification",
    "description": "Schema for defining hybrid automaton systems with modes, transitions, and configuration",
    "type": "object",
    "required": ["automaton", "config"],
    "additionalProperties": True,  # Allow init_state and other fields
    "properties": {
        "automaton": {
            "type": "object",
            "description": "The hybrid automaton definition with variables, modes, and transitions",
            "required": ["var", "mode", "edge"],
            "properties": {
                "var": {
                    "type": "string",
                    "description": "Comma-separated list of state variable names (e.g., 'x1' or 'x1, x2')",
                    "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*(\\s*,\\s*[a-zA-Z_][a-zA-Z0-9_]*)*$",
                    "minLength": 1,
                    "examples": ["x1", "x1, x2", "x, y, z"]
                },
                "input": {
                    "type": "string",
                    "description": "Comma-separated list of input variable names (e.g., 'u1' or 'u1, u2'). Empty string if no inputs.",
                    "pattern": "^([a-zA-Z_][a-zA-Z0-9_]*(\\s*,\\s*[a-zA-Z_][a-zA-Z0-9_]*)*)?$",
                    "default": "",
                    "examples": ["", "u1", "u1, u2"]
                },
                "mode": {
                    "type": "array",
                    "description": "List of discrete modes (operating regimes) in the automaton",
                    "minItems": 1,
                    "items": {
                        "$ref": "#/definitions/mode"
                    }
                },
                "edge": {
                    "type": "array",
                    "description": "List of discrete transitions between modes (can be empty [])",
                    "items": {
                        "$ref": "#/definitions/edge"
                    }
                }
            },
            "additionalProperties": False
        },
        "config": {
            "type": "object",
            "description": "Simulation and learning configuration parameters",
            "required": ["dt", "total_time"],
            "properties": {
                "dt": {
                    "type": "number",
                    "description": "Integration time step in seconds",
                    "exclusiveMinimum": 0,
                    "examples": [0.001, 0.01]
                },
                "total_time": {
                    "type": "number",
                    "description": "Total simulation duration in seconds",
                    "exclusiveMinimum": 0,
                    "examples": [10.0, 20.0]
                },
                "order": {
                    "type": "integer",
                    "description": "Order of the ODE system (1=first-order, 2=second-order, etc.)",
                    "minimum": 1,
                    "default": 1
                },
                "need_reset": {
                    "type": "boolean",
                    "description": "Whether state resets occur on mode transitions",
                    "default": False
                },
                "non_linear_items": {
                    "type": "string",
                    "description": "Nonlinear/cross terms in dynamics (e.g., 'x1[0]**3', 'x1[0]*x2[0]')",
                    "default": ""
                },
                "self_loop": {
                    "type": "boolean",
                    "description": "Whether self-loop transitions are allowed",
                    "default": False
                },
                "need_bias": {
                    "type": "boolean",
                    "description": "Whether to include constant term in ODEs"
                }
            },
            "additionalProperties": True  # Allow additional config options
        },
    },
    "definitions": {
        "mode": {
            "type": "object",
            "description": "A discrete mode (operating regime) with its continuous dynamics",
            "required": ["id", "eq"],
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Unique mode identifier (must be >= 1)",
                    "minimum": 1
                },
                "eq": {
                    "type": "string",
                    "description": "ODE equation(s) defining the continuous dynamics. Format: 'var[order] = expression'. Multiple equations separated by commas.",
                    "minLength": 1,
                    "examples": [
                        "x1[1] = -2 * x1[0] + u1",
                        "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1",
                        "x1[1] = x2[0], x2[1] = -9.8 + u1"
                    ]
                }
            },
            "additionalProperties": False
        },
        "edge": {
            "type": "object",
            "description": "A discrete transition (edge) between modes",
            "required": ["direction", "condition"],
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Transition direction in format 'source -> target' (e.g., '1 -> 2')",
                    "pattern": "^\\d+\\s*->\\s*\\d+$"
                },
                "condition": {
                    "type": "string",
                    "description": "Guard condition that triggers the transition. Use bare variable names (x1, not x1[0]).",
                    "minLength": 1,
                    "examples": [
                        "x1 > 0.5",
                        "x2 <= 4",
                        "x1 <= 0 and x2 > 1",
                        "abs(x1) >= 1.2"
                    ]
                },
                "reset": {
                    "type": "object",
                    "description": "Optional state reset map. Keys are variable names, values are arrays of reset expressions.",
                    "additionalProperties": {
                        "type": "array",
                        "description": "Reset values for each derivative order. Use '' to preserve current value.",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "number"}
                            ]
                        }
                    },
                    "examples": [
                        {"x1": [0], "x2": ["-0.9 * x2[0]"]},
                        {"x": ["", "x[1] * 0.95"]}
                    ]
                }
            },
            "additionalProperties": False
        }
    }
}



# Schema as formatted strings for inclusion in prompts
# HA_JSON_SCHEMA_STRING = json.dumps(HA_JSON_SCHEMA, indent=2)


# ==============================================================================
# Validation Functions
# ==============================================================================

def validate_ha_schema(ha_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an HA specification against the JSON Schema.

    Args:
        ha_dict: HA specification dictionary

    Returns:
        Dictionary with:
        - 'valid': bool - Whether the spec passes schema validation
        - 'errors': List[Dict] - List of error objects with 'path' and 'message'
    """
    if not HAS_JSONSCHEMA:
        return {
            'valid': False,
            'errors': [{'path': '', 'message': 'jsonschema library not installed'}]
        }

    errors = []
    validator = Draft7Validator(HA_JSON_SCHEMA)

    for error in sorted(validator.iter_errors(ha_dict), key=lambda e: list(e.path)):
        error_path = '.'.join(str(p) for p in error.path) if error.path else '(root)'
        errors.append({
            'path': error_path,
            'message': error.message,
            'schema_path': '.'.join(str(p) for p in error.schema_path)
        })

    return {
        'valid': len(errors) == 0,
        'errors': errors
    }


def validate_equation_format(eq_str: str, var_list: List[str], input_list: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate equation string syntax and structure.

    Args:
        eq_str: Equation string (e.g., "x1[2] = -0.5 * x1[1] + u1")
        var_list: List of declared variable names
        input_list: List of declared input variable names

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    if input_list is None:
        input_list = []

    errors = []

    # Split by comma for multiple equations
    equations = [eq.strip() for eq in eq_str.split(',') if eq.strip()]

    if not equations:
        return False, ["Equation string is empty"]

    defined_vars = set()

    for eq in equations:
        # Check for '=' sign
        if '=' not in eq:
            errors.append(f"Equation '{eq}' missing '=' sign")
            continue

        parts = eq.split('=')
        if len(parts) != 2:
            errors.append(f"Equation '{eq}' has invalid format (multiple '=' signs)")
            continue

        lhs, rhs = parts[0].strip(), parts[1].strip()

        # Check LHS format: should be var_name[order]
        lhs_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]$', lhs)
        if not lhs_match:
            errors.append(f"LHS '{lhs}' must be in format 'variable[order]' (e.g., x1[2])")
            continue

        var_name = lhs_match.group(1)
        order = int(lhs_match.group(2))

        if var_name not in var_list:
            errors.append(f"Variable '{var_name}' in equation not declared in 'var' field. Declared: {var_list}")
            continue

        if order < 1:
            errors.append(f"Derivative order must be >= 1, got {order} in '{lhs}'")
            continue

        defined_vars.add(var_name)

        # Check RHS can be parsed (basic syntax check)
        # Replace var[n] with placeholders and try to compile
        test_expr = rhs
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]'
        test_expr = re.sub(pattern, 'x_placeholder', test_expr)

        # Replace input variables
        for inp in input_list:
            test_expr = re.sub(rf'\b{re.escape(inp)}\b', 'u_placeholder', test_expr)

        try:
            compile(test_expr, '<string>', 'eval')
        except SyntaxError as e:
            errors.append(f"Equation '{eq}' has syntax error in RHS: {e}")

    # Check all variables have equations
    missing_vars = set(var_list) - defined_vars
    if missing_vars and len(equations) > 0:
        errors.append(f"Variables {missing_vars} declared but have no equations")

    return len(errors) == 0, errors


def validate_condition_format(condition: str, var_list: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate guard condition syntax.

    Args:
        condition: Condition string (e.g., "x1 > 0.5", "x1 <= 0 and x2 > 1")
        var_list: List of declared variable names

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    if not condition or not condition.strip():
        return False, ["Condition string is empty"]

    # Normalize: remove x[0] indexing if present
    test_condition = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\[0\]', r'\1', condition)

    # Replace variable names with True for syntax checking
    for var in var_list:
        test_condition = re.sub(rf'\b{re.escape(var)}\b', '1.0', test_condition)

    try:
        compile(test_condition, '<string>', 'eval')
    except SyntaxError as e:
        errors.append(f"Condition '{condition}' has syntax error: {e}")

    return len(errors) == 0, errors


def validate_direction_format(direction: str, mode_ids: List[int]) -> Tuple[bool, List[str]]:
    """
    Validate edge direction format and mode references.

    Args:
        direction: Direction string (e.g., "1 -> 2")
        mode_ids: List of valid mode IDs

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Check format
    match = re.match(r'^(\d+)\s*->\s*(\d+)$', direction)
    if not match:
        errors.append(f"Direction '{direction}' must be in format 'source -> target' (e.g., '1 -> 2')")
        return False, errors

    source, target = int(match.group(1)), int(match.group(2))

    if source not in mode_ids:
        errors.append(f"Source mode {source} in direction '{direction}' does not exist. Valid modes: {mode_ids}")

    if target not in mode_ids:
        errors.append(f"Target mode {target} in direction '{direction}' does not exist. Valid modes: {mode_ids}")

    return len(errors) == 0, errors


def validate_reset_format(reset: Dict[str, Any], var_list: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate reset map format.

    Args:
        reset: Reset dictionary (e.g., {"x1": [0], "x2": ["-0.9 * x2[0]"]})
        var_list: List of declared variable names

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    if not isinstance(reset, dict):
        return False, ["Reset must be an object/dictionary"]

    for var_name, reset_values in reset.items():
        if var_name not in var_list:
            errors.append(f"Reset variable '{var_name}' not declared in 'var' field")
            continue

        if not isinstance(reset_values, list):
            errors.append(f"Reset value for '{var_name}' must be an array, got {type(reset_values).__name__}")
            continue

        for i, val in enumerate(reset_values):
            if val == "" or val is None:
                continue  # Empty string means preserve current value
            if isinstance(val, (int, float)):
                continue  # Numeric constant is valid
            if isinstance(val, str):
                # Check if it's a valid expression
                try:
                    # Replace var[n] with placeholders
                    test_expr = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\[\d+\]', 'x_placeholder', val)
                    compile(test_expr, '<string>', 'eval')
                except SyntaxError as e:
                    errors.append(f"Reset expression '{val}' for {var_name}[{i}] has syntax error: {e}")
            else:
                errors.append(f"Reset value for {var_name}[{i}] must be number or string expression, got {type(val).__name__}")

    return len(errors) == 0, errors


def validate_ha_with_schema(ha_input: Any, auto_extract: bool = True) -> Dict[str, Any]:
    """
    Comprehensive validation of HA specification using JSON Schema and semantic checks.

    This is the main entry point for validation. It performs:
    1. JSON Schema validation (structure, types, patterns)
    2. Semantic validation (equation syntax, condition syntax, mode references)

    Args:
        ha_input: HA specification (dict or JSON string)
        auto_extract: If True, attempt to extract dict from string input

    Returns:
        Dictionary with:
        - 'valid': bool - Overall validation result
        - 'schema_valid': bool - Schema validation result
        - 'semantic_valid': bool - Semantic validation result
        - 'errors': List[Dict] - All errors with 'path', 'message', 'type'
        - 'ha_dict': Dict or None - The parsed/extracted HA dictionary
        - 'summary': str - Human-readable summary
    """
    result = {
        'valid': False,
        'schema_valid': False,
        'semantic_valid': False,
        'errors': [],
        'ha_dict': None,
        'summary': ''
    }

    # Step 1: Extract dictionary from input
    if isinstance(ha_input, dict):
        ha_dict = ha_input
    elif isinstance(ha_input, str) and auto_extract:
        try:
            ha_dict = json.loads(ha_input)
        except json.JSONDecodeError:
            # Try to extract from markdown code blocks
            code_block_pattern = r'```(?:json|python)?\s*([\s\S]*?)\s*```'
            matches = re.findall(code_block_pattern, ha_input)
            if matches:
                try:
                    # Convert Python booleans to JSON
                    text = matches[-1].replace("True", "true").replace("False", "false").replace("None", "null")
                    ha_dict = json.loads(text)
                except json.JSONDecodeError as e:
                    result['errors'].append({
                        'path': '(extraction)',
                        'message': f'Failed to parse JSON from code block: {e}',
                        'type': 'extraction'
                    })
                    result['summary'] = 'Failed to extract valid JSON from input'
                    return result
            else:
                result['errors'].append({
                    'path': '(extraction)',
                    'message': 'Input is not valid JSON and no code block found',
                    'type': 'extraction'
                })
                result['summary'] = 'Failed to extract valid JSON from input'
                return result
    else:
        result['errors'].append({
            'path': '(input)',
            'message': f'Invalid input type: {type(ha_input).__name__}, expected dict or JSON string',
            'type': 'input'
        })
        result['summary'] = 'Invalid input type'
        return result

    result['ha_dict'] = ha_dict

    # Step 2: JSON Schema validation
    schema_result = validate_ha_schema(ha_dict)
    result['schema_valid'] = schema_result['valid']

    for error in schema_result['errors']:
        result['errors'].append({
            'path': error['path'],
            'message': error['message'],
            'type': 'schema'
        })

    # Step 3: Semantic validation (only if basic structure is present)
    semantic_errors = []

    if 'automaton' in ha_dict and isinstance(ha_dict['automaton'], dict):
        automaton = ha_dict['automaton']

        # Extract variable and input lists
        var_str = automaton.get('var', '')
        var_list = [v.strip() for v in var_str.split(',') if v.strip()] if var_str else []

        input_str = automaton.get('input', '')
        input_list = [v.strip() for v in input_str.split(',') if v.strip()] if input_str else []

        # Collect mode IDs
        mode_ids = []
        modes = automaton.get('mode', [])
        if isinstance(modes, list):
            for mode in modes:
                if isinstance(mode, dict) and 'id' in mode:
                    mode_ids.append(mode['id'])

        # Validate each mode's equation
        if isinstance(modes, list):
            for i, mode in enumerate(modes):
                if isinstance(mode, dict) and 'eq' in mode:
                    eq_valid, eq_errors = validate_equation_format(mode['eq'], var_list, input_list)
                    for err in eq_errors:
                        semantic_errors.append({
                            'path': f'automaton.mode[{i}].eq',
                            'message': err,
                            'type': 'semantic'
                        })

        # Validate each edge
        edges = automaton.get('edge', [])
        if isinstance(edges, list):
            for i, edge in enumerate(edges):
                if isinstance(edge, dict):
                    # Validate direction
                    if 'direction' in edge:
                        dir_valid, dir_errors = validate_direction_format(edge['direction'], mode_ids)
                        for err in dir_errors:
                            semantic_errors.append({
                                'path': f'automaton.edge[{i}].direction',
                                'message': err,
                                'type': 'semantic'
                            })

                    # Validate condition
                    if 'condition' in edge:
                        cond_valid, cond_errors = validate_condition_format(edge['condition'], var_list)
                        for err in cond_errors:
                            semantic_errors.append({
                                'path': f'automaton.edge[{i}].condition',
                                'message': err,
                                'type': 'semantic'
                            })

                    # Validate reset
                    if 'reset' in edge:
                        reset_valid, reset_errors = validate_reset_format(edge['reset'], var_list)
                        for err in reset_errors:
                            semantic_errors.append({
                                'path': f'automaton.edge[{i}].reset',
                                'message': err,
                                'type': 'semantic'
                            })

    result['errors'].extend(semantic_errors)
    result['semantic_valid'] = len(semantic_errors) == 0

    # Overall result
    result['valid'] = result['schema_valid'] and result['semantic_valid']

    # Generate summary
    if result['valid']:
        automaton = ha_dict.get('automaton', {})
        num_modes = len(automaton.get('mode', []))
        num_edges = len(automaton.get('edge', []))
        result['summary'] = f"Valid HA specification with {num_modes} mode(s) and {num_edges} edge(s)"
    else:
        schema_errors = sum(1 for e in result['errors'] if e['type'] == 'schema')
        semantic_errors = sum(1 for e in result['errors'] if e['type'] == 'semantic')
        result['summary'] = f"Invalid: {schema_errors} schema error(s), {semantic_errors} semantic error(s)"

    return result


def format_validation_errors(result: Dict[str, Any], verbose: bool = True) -> str:
    """
    Format validation result as a human-readable string.

    Args:
        result: Result from validate_ha_with_schema()
        verbose: If True, include all error details

    Returns:
        Formatted string describing validation result
    """
    lines = []

    if result['valid']:
        lines.append("✅ **VALID**: HA specification passed all validation checks")
        lines.append(f"   {result['summary']}")
    else:
        lines.append("❌ **INVALID**: HA specification has errors")
        lines.append(f"   {result['summary']}")

        if verbose and result['errors']:
            lines.append("\n**Errors:**")

            # Group by type
            schema_errors = [e for e in result['errors'] if e['type'] == 'schema']
            semantic_errors = [e for e in result['errors'] if e['type'] == 'semantic']
            other_errors = [e for e in result['errors'] if e['type'] not in ('schema', 'semantic')]

            if other_errors:
                lines.append("\n  *Extraction/Input Errors:*")
                for err in other_errors:
                    lines.append(f"    - {err['message']}")

            if schema_errors:
                lines.append("\n  *Schema Errors (structure/type issues):*")
                for err in schema_errors[:10]:  # Limit to first 10
                    lines.append(f"    - [{err['path']}] {err['message']}")
                if len(schema_errors) > 10:
                    lines.append(f"    ... and {len(schema_errors) - 10} more")

            if semantic_errors:
                lines.append("\n  *Semantic Errors (logic/syntax issues):*")
                for err in semantic_errors[:10]:
                    lines.append(f"    - [{err['path']}] {err['message']}")
                if len(semantic_errors) > 10:
                    lines.append(f"    ... and {len(semantic_errors) - 10} more")

    return '\n'.join(lines)


# ==============================================================================
# Schema as prompt-ready string
# ==============================================================================

def get_schema_for_prompt(include_definitions: bool = True) -> str:
    """
    Get the JSON Schema as a formatted string suitable for inclusion in LLM prompts.

    Args:
        include_definitions: If True, include the full schema with definitions (without $schema/$id)

    Returns:
        Formatted JSON Schema string
    """
    if include_definitions:
        return json.dumps(HA_JSON_SCHEMA, indent=2)
    else:
        # Simplified version without $ref definitions
        simplified = {k: v for k, v in HA_JSON_SCHEMA.items() if k not in ("$schema", "$id")}
        return json.dumps(simplified, indent=2)


# ==============================================================================
# Test Cases
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing JSON Schema Validation for Hybrid Automaton Specifications")
    print("=" * 70)

    # Test 1: Valid single-mode specification
    print("\n--- Test 1: Valid single-mode Duffing oscillator ---")
    valid_spec_1 = {
        "automaton": {
            "var": "x1",
            "input": "u1",
            "mode": [
                {"id": 1, "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1"}
            ],
            "edge": []
        },
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "dim": 2
        }
    }
    result = validate_ha_with_schema(valid_spec_1)
    print(format_validation_errors(result))
    assert result['valid'], "Test 1 should pass"

    # Test 2: Valid multi-mode specification with edges
    print("\n--- Test 2: Valid two-mode thermostat system ---")
    valid_spec_2 = {
        "automaton": {
            "var": "x1",
            "input": "",
            "mode": [
                {"id": 1, "eq": "x1[1] = -0.1 * x1[0] + 5"},
                {"id": 2, "eq": "x1[1] = -0.1 * x1[0] - 3"}
            ],
            "edge": [
                {"direction": "1 -> 2", "condition": "x1 >= 25"},
                {"direction": "2 -> 1", "condition": "x1 <= 15"}
            ]
        },
        "config": {
            "dt": 0.01,
            "total_time": 20.0
        }
    }
    result = validate_ha_with_schema(valid_spec_2)
    print(format_validation_errors(result))
    assert result['valid'], "Test 2 should pass"

    # Test 3: Valid bouncing ball with reset
    print("\n--- Test 3: Valid bouncing ball with reset ---")
    valid_spec_3 = {
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
                    "reset": {"x1": [0], "x2": ["-0.9 * x2[0]"]}
                }
            ]
        },
        "config": {
            "dt": 0.01,
            "total_time": 10.0,
            "need_reset": True
        }
    }
    result = validate_ha_with_schema(valid_spec_3)
    print(format_validation_errors(result))
    assert result['valid'], "Test 3 should pass"

    # Test 4: Missing required field 'automaton'
    print("\n--- Test 4: Missing 'automaton' field ---")
    invalid_spec_1 = {
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    result = validate_ha_with_schema(invalid_spec_1)
    print(format_validation_errors(result))
    assert not result['valid'], "Test 4 should fail"
    assert any('automaton' in str(e) for e in result['errors']), "Should mention missing automaton"

    # Test 5: Float mode ID type (JSON Schema may accept 1.0 as integer)
    # Note: Some JSON Schema validators accept 1.0 as valid integer
    # The ha_spec_validator auto-fixes this case
    print("\n--- Test 5: Float mode ID (handled by auto-fix) ---")
    invalid_spec_2 = {
        "automaton": {
            "var": "x1",
            "input": "",
            "mode": [{"id": 1.0, "eq": "x1[1] = -x1[0]"}],  # Float ID
            "edge": []
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    result = validate_ha_with_schema(invalid_spec_2)
    print(format_validation_errors(result))
    # Schema may pass; validate_and_fix_ha_spec() auto-converts float to int
    print(f"Note: JSON Schema {'accepted' if result['valid'] else 'rejected'} 1.0 as mode ID")

    # Test 6: Invalid direction format
    print("\n--- Test 6: Invalid direction format ---")
    invalid_spec_3 = {
        "automaton": {
            "var": "x1",
            "input": "",
            "mode": [{"id": 1, "eq": "x1[1] = -x1[0]"}],
            "edge": [{"direction": "1->2", "condition": "x1 > 0"}]  # Missing spaces
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    result = validate_ha_with_schema(invalid_spec_3)
    print(format_validation_errors(result))
    # Note: Schema allows flexible spacing, but semantic check validates mode existence

    # Test 7: Equation with undeclared variable
    print("\n--- Test 7: Equation with undeclared variable ---")
    invalid_spec_4 = {
        "automaton": {
            "var": "x1",
            "input": "",
            "mode": [{"id": 1, "eq": "x2[1] = -x1[0]"}],  # x2 not declared
            "edge": []
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    result = validate_ha_with_schema(invalid_spec_4)
    print(format_validation_errors(result))
    assert not result['valid'], "Test 7 should fail"

    # Test 8: Empty mode array
    print("\n--- Test 8: Empty mode array ---")
    invalid_spec_5 = {
        "automaton": {
            "var": "x1",
            "input": "",
            "mode": [],  # Empty
            "edge": []
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    result = validate_ha_with_schema(invalid_spec_5)
    print(format_validation_errors(result))
    assert not result['valid'], "Test 8 should fail"

    # Test 9: JSON string input
    print("\n--- Test 9: JSON string input ---")
    json_string = '{"automaton": {"var": "x1", "input": "", "mode": [{"id": 1, "eq": "x1[1] = -x1[0]"}], "edge": []}, "config": {"dt": 0.001, "total_time": 10.0}}'
    result = validate_ha_with_schema(json_string)
    print(format_validation_errors(result))
    assert result['valid'], "Test 9 should pass"

    # Test 10: Markdown code block input
    print("\n--- Test 10: Markdown code block input ---")
    markdown_input = '''Here is the specification:
```json
{"automaton": {"var": "x1", "input": "", "mode": [{"id": 1, "eq": "x1[1] = -x1[0]"}], "edge": []}, "config": {"dt": 0.001, "total_time": 10.0}}
```
'''
    result = validate_ha_with_schema(markdown_input)
    print(format_validation_errors(result))
    assert result['valid'], "Test 10 should pass"

    # Test 11: Edge referencing non-existent mode
    print("\n--- Test 11: Edge referencing non-existent mode ---")
    invalid_spec_6 = {
        "automaton": {
            "var": "x1",
            "input": "",
            "mode": [{"id": 1, "eq": "x1[1] = -x1[0]"}],
            "edge": [{"direction": "1 -> 2", "condition": "x1 > 0"}]  # Mode 2 doesn't exist
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    result = validate_ha_with_schema(invalid_spec_6)
    print(format_validation_errors(result))
    assert not result['valid'], "Test 11 should fail"

    # Test 12: Invalid condition syntax
    print("\n--- Test 12: Invalid condition syntax ---")
    invalid_spec_7 = {
        "automaton": {
            "var": "x1",
            "input": "",
            "mode": [{"id": 1, "eq": "x1[1] = -x1[0]"}, {"id": 2, "eq": "x1[1] = x1[0]"}],
            "edge": [{"direction": "1 -> 2", "condition": "x1 >> 0"}]  # Invalid operator
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    result = validate_ha_with_schema(invalid_spec_7)
    print(format_validation_errors(result))
    # Note: >> is valid Python (bitwise shift), so this may pass syntax check

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
