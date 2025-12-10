"""
Hybrid Automaton Specification Validation Tool

This tool allows the agent to validate its HA specification before submitting
the final answer. It checks syntax and structure, and attempts to fix common issues.

Uses both traditional validation and JSON Schema validation for comprehensive checking.
"""

import json
from typing import Any, Dict
from smolagents.default_tools import Tool

# Support both relative imports (when used as package) and absolute imports (when run directly)
try:
    from .ha_spec_validator import validate_and_fix_ha_spec, validate_equation_syntax, validate_condition_syntax
except ImportError:
    from ha_spec_validator import validate_and_fix_ha_spec, validate_equation_syntax, validate_condition_syntax

# Import JSON Schema validation
try:
    from .ha_json_schema import validate_ha_with_schema, format_validation_errors, HAS_JSONSCHEMA
except ImportError:
    try:
        from ha_json_schema import validate_ha_with_schema, format_validation_errors, HAS_JSONSCHEMA
    except ImportError:
        HAS_JSONSCHEMA = False
        validate_ha_with_schema = None
        format_validation_errors = None


class ValidateHASpecTool(Tool):
    """Tool for validating and fixing Hybrid Automaton specifications before submission."""

    name = "validate_hybrid_automaton_specification"
    description = (
        "**Hybrid Automaton Specification Validator**\n\n"
        "Use this tool to validate your HA specification BEFORE submitting your final answer.\n"
        "The tool checks for syntax errors and common issues that would cause evaluation to fail.\n\n"
        "**WHAT IT CHECKS:**\n"
        "1. Required keys: 'automaton' (var, input, mode, edge) and 'config' (dt, total_time)\n"
        "2. Mode structure: id (must be integer), eq (must be valid ODE syntax)\n"
        "3. Edge structure: direction (must be 'source -> target' format), condition (must use bare variable names)\n"
        "4. Equation syntax: validates that ODEs can be parsed (e.g., 'x1[2] = -0.5 * x1[1] + u1')\n"
        "5. Condition syntax: validates guard conditions (e.g., 'x1 > 0' NOT 'x1[0] > 0')\n\n"
        "**AUTO-FIX CAPABILITIES:**\n"
        "- Converts float mode IDs to integers (1.0 → 1)\n"
        "- Fixes direction format ('1->2' → '1 -> 2')\n"
        "- Normalizes condition format (x1[0] → x1, both are valid but bare names preferred)\n"
        "- Adds missing 'edge' or 'input' keys with defaults\n\n"
        "**HOW TO USE:**\n"
        "Pass your HA specification as a JSON string. The tool returns:\n"
        "- VALID: Your specification is ready for submission\n"
        "- FIXED: Issues were auto-corrected, use the returned fixed specification\n"
        "- INVALID: Critical errors found that cannot be auto-fixed\n\n"
        "**IMPORTANT:** Always call this tool before submitting your final answer!"
    )
    inputs = {
        "ha_spec_json": {
            "type": "string",
            "description": (
                "Your complete Hybrid Automaton specification as a JSON string. "
                "Must include 'automaton' and 'config' sections. "
                "Use JSON format: true/false (not True/False), double quotes. Example:\n"
                '{"automaton": {"var": "x1", "input": "u1", "mode": [{"id": 1, "eq": "x1[2] = -0.5 * x1[1]"}], "edge": []}, '
                '"config": {"dt": 0.001, "total_time": 10.0, "need_reset": false}}'
            )
        },
    }
    output_type = "string"

    def __init__(self, worker_agent=None):
        super().__init__()
        self.worker_agent = worker_agent  # Reference to main agent (optional)

    def forward(self, ha_spec_json: str) -> str:  # type: ignore[override]
        """
        Validate and optionally fix the HA specification.

        Args:
            ha_spec_json: HA specification as a JSON string

        Returns:
            Validation result with status, messages, and fixed specification if applicable
        """

        # Step 1: Run traditional validation and auto-fix
        ha_dict, is_valid, messages = validate_and_fix_ha_spec(ha_spec_json, auto_fix=True)

        # Step 2: Run JSON Schema validation if available
        schema_result = None
        if HAS_JSONSCHEMA and validate_ha_with_schema is not None and ha_dict is not None:
            schema_result = validate_ha_with_schema(ha_dict, auto_extract=False)

        # Step 3: Build response
        response_parts = []

        if ha_dict is None:
            response_parts.append("❌ **STATUS: INVALID - EXTRACTION FAILED**")
            response_parts.append("\nCould not extract a valid dictionary from your input.")
            response_parts.append("\n**Errors:**")
            for msg in messages:
                response_parts.append(f"  - {msg}")
            response_parts.append("\n**Action Required:** Please provide a valid Python dictionary with 'automaton' and 'config' keys.")
            return "\n".join(response_parts)

        # Check if any fixes were applied
        fixes_applied = [msg for msg in messages if msg.startswith("Fixed:")]
        errors_remaining = [msg for msg in messages if msg.startswith("Remaining error:") or msg.startswith("Error:")]

        # Combine validation results
        schema_valid = schema_result['valid'] if schema_result else True
        overall_valid = is_valid and schema_valid

        if overall_valid:
            if fixes_applied:
                response_parts.append("✅ **STATUS: FIXED - USE CORRECTED SPECIFICATION**")
                response_parts.append("\nYour specification had issues that were auto-corrected.")
                response_parts.append("\n**Fixes Applied:**")
                for fix in fixes_applied:
                    response_parts.append(f"  - {fix.replace('Fixed: ', '')}")
            else:
                response_parts.append("✅ **STATUS: VALID - READY FOR SUBMISSION**")
                response_parts.append("\nYour HA specification passed all validation checks.")

            # Additional syntax validation details
            response_parts.append("\n**Validation Summary:**")
            automaton = ha_dict.get('automaton', {})
            var_list = [v.strip() for v in automaton.get('var', '').split(',') if v.strip()]
            input_list = [v.strip() for v in automaton.get('input', '').split(',') if v.strip()]

            response_parts.append(f"  - Variables: {var_list}")
            response_parts.append(f"  - Inputs: {input_list}")
            response_parts.append(f"  - Modes: {len(automaton.get('mode', []))}")
            response_parts.append(f"  - Edges: {len(automaton.get('edge', []))}")

            if schema_result:
                response_parts.append(f"  - Schema validation: ✅ Passed")

            # Provide the corrected specification
            response_parts.append("\n**Corrected Specification (use this for final answer):**")
            response_parts.append(f"```json\n{json.dumps(ha_dict, indent=2)}\n```")

        else:
            response_parts.append("❌ **STATUS: INVALID - CRITICAL ERRORS FOUND**")
            response_parts.append("\nYour specification has errors that could not be auto-fixed.")

            if fixes_applied:
                response_parts.append("\n**Fixes Applied (partial):**")
                for fix in fixes_applied:
                    response_parts.append(f"  - {fix.replace('Fixed: ', '')}")

            response_parts.append("\n**Remaining Errors (must fix manually):**")
            for err in errors_remaining:
                response_parts.append(f"  - {err.replace('Remaining error: ', '').replace('Error: ', '')}")

            # Add schema validation errors if any
            if schema_result and not schema_result['valid']:
                response_parts.append("\n**JSON Schema Validation Errors:**")
                for err in schema_result.get('errors', [])[:5]:  # Limit to first 5
                    response_parts.append(f"  - [{err['path']}] {err['message']}")
                if len(schema_result.get('errors', [])) > 5:
                    response_parts.append(f"  ... and {len(schema_result['errors']) - 5} more errors")

            # Provide detailed error analysis
            response_parts.append("\n**Common Issues & Solutions:**")

            # Check for specific error patterns and provide targeted help
            all_error_text = " ".join(errors_remaining).lower()
            if schema_result:
                all_error_text += " " + " ".join(str(e) for e in schema_result.get('errors', []))

            if "mode" in all_error_text and ("id" in all_error_text or "missing" in all_error_text or "integer" in all_error_text):
                response_parts.append("  📌 Mode ID issue: Ensure each mode has 'id': 1, 'id': 2, etc. (integers, not floats like 1.0)")

            if "eq" in all_error_text or "equation" in all_error_text:
                response_parts.append("  📌 Equation syntax: Use format 'x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1'")
                response_parts.append("     - LHS: variable[order] (highest derivative)")
                response_parts.append("     - RHS: valid Python expression with x1[0], x1[1], u1, etc.")

            if "condition" in all_error_text:
                response_parts.append("  📌 Guard condition: Use bare variable names (x1 > 0), NOT indexed (x1[0] > 0)")

            if "direction" in all_error_text:
                response_parts.append("  📌 Edge direction: Use exact format '1 -> 2' (with spaces around arrow)")

            if "minItems" in all_error_text or "empty" in all_error_text.lower():
                response_parts.append("  📌 Empty array: 'mode' must have at least one mode defined")

            # Show partial spec for reference
            if ha_dict:
                response_parts.append("\n**Partially Parsed Specification:**")
                response_parts.append(f"```json\n{json.dumps(ha_dict, indent=2)}\n```")

        return "\n".join(response_parts)


# Quick validation function for programmatic use
def quick_validate(ha_spec: Any) -> tuple[bool, str]:
    """
    Quick validation check for HA specification.
    
    Args:
        ha_spec: HA specification (dict or JSON string)
        
    Returns:
        Tuple of (is_valid, brief_message)
    """
    ha_dict, is_valid, messages = validate_and_fix_ha_spec(ha_spec, auto_fix=True)
    
    if is_valid:
        return True, "Specification is valid"
    else:
        errors = [m for m in messages if "error" in m.lower()]
        return False, f"Invalid: {'; '.join(errors[:3])}"


if __name__ == "__main__":
    # Test the tool
    tool = ValidateHASpecTool()
    
    # Test 1: Valid specification
    print("=" * 60)
    print("Test 1: Valid specification")
    print("=" * 60)
    valid_spec = json.dumps({
        "automaton": {
            "var": "x1",
            "input": "u1",
            "mode": [{"id": 1, "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1"}],
            "edge": []
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    })
    print(tool.forward(valid_spec))
    
    # Test 2: Fixable specification
    print("\n" + "=" * 60)
    print("Test 2: Fixable specification (float id, wrong direction format, x1[0] in condition)")
    print("=" * 60)
    fixable_spec = json.dumps({
        "automaton": {
            "var": "x1, x2",
            "input": "u1",
            "mode": [{"id": 1.0, "eq": "x1[1] = x2[0], x2[1] = -9.8 + u1"}],
            "edge": [{"direction": "1->1", "condition": "x1[0] <= 0", "reset": {"x2": ["-0.9 * x2[0]"]}}]
        },
        "config": {"dt": 0.001}
    })
    print(tool.forward(fixable_spec))
    
    # Test 3: Invalid specification
    print("\n" + "=" * 60)
    print("Test 3: Invalid specification (missing mode)")
    print("=" * 60)
    invalid_spec = json.dumps({
        "automaton": {
            "var": "x1",
            "edge": []
        },
        "config": {"dt": 0.001, "total_time": 10.0}
    })
    print(tool.forward(invalid_spec))

