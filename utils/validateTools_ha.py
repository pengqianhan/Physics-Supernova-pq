"""
Hybrid Automaton Specification Validation Tool

This tool allows the agent to validate its HA specification before submitting
the final answer. It checks syntax and structure, and attempts to fix common issues.

Uses JSON Schema validation as the primary validation mechanism.
"""

import json
from typing import Any, Dict
from smolagents.default_tools import Tool

# Support both relative imports (when used as package) and absolute imports (when run directly)
try:
    from .ha_spec_validator import extract_and_fix_ha_spec
except ImportError:
    from ha_spec_validator import extract_and_fix_ha_spec

# Import JSON Schema validation (primary validation)
try:
    from .ha_json_schema import validate_ha_with_schema, format_validation_errors, HAS_JSONSCHEMA
except ImportError:
    try:
        from ha_json_schema import validate_ha_with_schema, format_validation_errors, HAS_JSONSCHEMA
    except ImportError:
        HAS_JSONSCHEMA = False
        validate_ha_with_schema = None
        format_validation_errors = None

# Import Python class validation (for new class-based format)
try:
    from .ha_class_validator import (
        extract_python_class_from_text,
        validate_python_class_syntax,
        auto_fix_common_class_issues
    )
except ImportError:
    try:
        from ha_class_validator import (
            extract_python_class_from_text,
            validate_python_class_syntax,
            auto_fix_common_class_issues
        )
    except ImportError:
        # Fallback if class validator not available
        extract_python_class_from_text = None
        validate_python_class_syntax = None
        auto_fix_common_class_issues = None


class ValidateHASpecTool(Tool):
    """Tool for validating and fixing Hybrid Automaton specifications before submission."""

    name = "validate_hybrid_automaton_specification"
    description = (
        "**Hybrid Automaton Specification Validator**\n\n"
        "Use this tool to validate your HA specification BEFORE submitting your final answer.\n"
        "Supports BOTH Python class format and JSON format.\n\n"
        "**WHAT IT CHECKS:**\n"
        "For JSON format:\n"
        "1. Required keys: 'automaton' (var, input, mode, edge) and 'config' (dt, total_time)\n"
        "2. Mode structure: id (must be integer), eq (must be valid ODE syntax)\n"
        "3. Edge structure: direction (must be 'source -> target' format), condition (must use bare variable names)\n"
        "4. Equation syntax: validates that ODEs can be parsed (e.g., 'x1[2] = -0.5 * x1[1] + u1')\n"
        "5. Condition syntax: validates guard conditions (e.g., 'x1 > 0' NOT 'x1[0] > 0')\n\n"
        "For Python class format:\n"
        "1. Class named 'HybridAutomaton' exists\n"
        "2. Required methods: __init__, num_modes, mode_dynamics, guard_condition, reset_map, to_json\n"
        "3. self.params attribute defined (list of numerical parameters)\n"
        "4. Python syntax is valid\n\n"
        "**AUTO-FIX CAPABILITIES:**\n"
        "JSON: Converts float mode IDs to integers, fixes direction format, normalizes conditions\n"
        "Python class: Converts params tuple to list, adds missing to_json() stub\n\n"
        "**HOW TO USE:**\n"
        "Pass your HA specification as either:\n"
        "- JSON string: {\"automaton\": {...}, \"config\": {...}}\n"
        "- Python class code: class HybridAutomaton: ...\n"
        "The tool returns:\n"
        "- VALID: Your specification is ready for submission\n"
        "- FIXED: Issues were auto-corrected, use the returned fixed specification\n"
        "- INVALID: Critical errors found that cannot be auto-fixed\n\n"
        "**IMPORTANT:** Always call this tool before submitting your final answer!"
    )
    inputs = {
        "ha_spec_json": {
            "type": "string",
            "description": (
                "Your complete Hybrid Automaton specification as EITHER:\n"
                "1. JSON string: {\"automaton\": {...}, \"config\": {...}}, OR\n"
                "2. Python class code: class HybridAutomaton: ... (with params array)\n\n"
                "JSON example:\n"
                '{"automaton": {"var": "x1", "input": "u1", "mode": [{"id": 1, "eq": "x1[2] = -0.5 * x1[1]"}], "edge": []}, '
                '"config": {"dt": 0.001, "total_time": 10.0}}\n\n'
                "Python class example:\n"
                "class HybridAutomaton:\n"
                "    def __init__(self):\n"
                "        self.params = [-0.5, -5.0, 1.0, ...]\n"
                "        self.var = \"x1\"\n"
                "    def to_json(self): ..."
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

        Supports both JSON format and Python class format.
        Automatically detects format and routes to appropriate validator.

        Args:
            ha_spec_json: HA specification as either JSON string or Python class code

        Returns:
            Validation result with status, messages, and fixed specification if applicable
        """

        # STEP 0: Detect format (Python class vs JSON)
        is_python_class = 'class HybridAutomaton' in ha_spec_json

        if is_python_class:
            # Route to Python class validation
            return self._validate_python_class(ha_spec_json)
        else:
            # Route to JSON validation (original logic)
            return self._validate_json_format(ha_spec_json)

    def _validate_python_class(self, class_code: str) -> str:
        """Validate Python class format."""
        response_parts = []

        # Check if class validator is available
        if extract_python_class_from_text is None or validate_python_class_syntax is None:
            response_parts.append("❌ **ERROR: Python class validator not available**")
            response_parts.append("\nThe ha_class_validator module is not installed.")
            response_parts.append("Please use JSON format instead.")
            return "\n".join(response_parts)

        # Step 1: Extract class from markdown (if present)
        extracted_class = extract_python_class_from_text(class_code)
        if extracted_class is None:
            # Try using input directly
            extracted_class = class_code

        # Step 2: Validate syntax
        is_valid, errors = validate_python_class_syntax(extracted_class)

        if is_valid:
            response_parts.append("✅ **STATUS: VALID - READY FOR SUBMISSION**")
            response_parts.append("\nYour Python class passed all validation checks.")

            # Extract metadata
            try:
                from utils.ha_class_to_json import extract_metadata_from_class
                metadata = extract_metadata_from_class(extracted_class)

                response_parts.append("\n**Validation Summary:**")
                response_parts.append(f"  - Variables: {metadata['var']}")
                response_parts.append(f"  - Inputs: {metadata['input']}")
                response_parts.append(f"  - Modes: {metadata['num_modes']}")
                response_parts.append(f"  - Params: {len(metadata['params'])} parameters")
                response_parts.append(f"  - Python syntax: ✅ Valid")
            except Exception:
                response_parts.append("\n**Validation Summary:**")
                response_parts.append(f"  - Python syntax: ✅ Valid")
                response_parts.append(f"  - Required methods: ✅ Present")

            response_parts.append("\n**Validated Class Code:**")
            response_parts.append(f"```python\n{extracted_class}\n```")

        else:
            # Try auto-fix
            fixed_class = auto_fix_common_class_issues(extracted_class) if auto_fix_common_class_issues else extracted_class

            # Re-validate after fixing
            is_valid_after_fix, errors_after_fix = validate_python_class_syntax(fixed_class)

            if is_valid_after_fix:
                response_parts.append("✅ **STATUS: FIXED - USE CORRECTED CLASS**")
                response_parts.append("\nYour class had issues that were auto-corrected.")

                response_parts.append("\n**Fixes Applied:**")
                if fixed_class != extracted_class:
                    response_parts.append("  - Auto-fixed common issues")

                response_parts.append("\n**Corrected Class Code (use this):**")
                response_parts.append(f"```python\n{fixed_class}\n```")

            else:
                response_parts.append("❌ **STATUS: INVALID - CRITICAL ERRORS FOUND**")
                response_parts.append("\nYour Python class has errors that could not be auto-fixed.")

                response_parts.append("\n**Validation Errors:**")
                for err in errors_after_fix[:10]:  # Limit to first 10
                    response_parts.append(f"  - {err}")
                if len(errors_after_fix) > 10:
                    response_parts.append(f"  ... and {len(errors_after_fix) - 10} more errors")

                response_parts.append("\n**Common Issues & Solutions:**")
                response_parts.append("  📌 Missing methods: Ensure __init__, num_modes, mode_dynamics, guard_condition, reset_map, to_json are defined")
                response_parts.append("  📌 Params format: Use self.params = [1.0, 2.0, ...] (list, not tuple)")
                response_parts.append("  📌 Method signatures: Check that methods have correct number of arguments")

                response_parts.append("\n**Partially Parsed Class:**")
                response_parts.append(f"```python\n{extracted_class[:500]}...\n```")

        return "\n".join(response_parts)

    def _validate_json_format(self, ha_spec_json: str) -> str:
        """Validate JSON format (original logic)."""
        # Step 1: Extract and auto-fix the specification
        ha_dict, fix_messages = extract_and_fix_ha_spec(ha_spec_json, auto_fix=True)

        # Build response
        response_parts = []

        if ha_dict is None:
            response_parts.append("❌ **STATUS: INVALID - EXTRACTION FAILED**")
            response_parts.append("\nCould not extract a valid dictionary from your input.")
            response_parts.append("\n**Errors:**")
            for msg in fix_messages:
                response_parts.append(f"  - {msg}")
            response_parts.append("\n**Action Required:** Please provide a valid Python dictionary with 'automaton' and 'config' keys.")
            return "\n".join(response_parts)

        # Step 2: Run JSON Schema validation (primary validation)
        if not HAS_JSONSCHEMA or validate_ha_with_schema is None:
            response_parts.append("⚠️ **WARNING: jsonschema library not installed**")
            response_parts.append("\nCannot perform full validation. Install with: pip install jsonschema")
            response_parts.append(f"\n**Extracted Specification:**\n```json\n{json.dumps(ha_dict, indent=2)}\n```")
            return "\n".join(response_parts)

        schema_result = validate_ha_with_schema(ha_dict, auto_extract=False)

        # Check if any fixes were applied
        fixes_applied = [msg for msg in fix_messages if msg.startswith("Fixed:")]

        # Use schema validation result as the primary validation result
        is_valid = schema_result['valid']

        if is_valid:
            if fixes_applied:
                response_parts.append("✅ **STATUS: FIXED - USE CORRECTED SPECIFICATION**")
                response_parts.append("\nYour specification had issues that were auto-corrected.")
                response_parts.append("\n**Fixes Applied:**")
                for fix in fixes_applied:
                    response_parts.append(f"  - {fix.replace('Fixed: ', '')}")
            else:
                response_parts.append("✅ **STATUS: VALID - READY FOR SUBMISSION**")
                response_parts.append("\nYour HA specification passed all validation checks.")

            # Validation summary
            response_parts.append("\n**Validation Summary:**")
            automaton = ha_dict.get('automaton', {})
            var_list = [v.strip() for v in automaton.get('var', '').split(',') if v.strip()]
            input_list = [v.strip() for v in automaton.get('input', '').split(',') if v.strip()]

            response_parts.append(f"  - Variables: {var_list}")
            response_parts.append(f"  - Inputs: {input_list}")
            response_parts.append(f"  - Modes: {len(automaton.get('mode', []))}")
            response_parts.append(f"  - Edges: {len(automaton.get('edge', []))}")
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

            # Show validation errors from JSON Schema
            response_parts.append("\n**Validation Errors:**")
            for err in schema_result.get('errors', [])[:10]:  # Limit to first 10
                response_parts.append(f"  - [{err['path']}] {err['message']}")
            if len(schema_result.get('errors', [])) > 10:
                response_parts.append(f"  ... and {len(schema_result['errors']) - 10} more errors")

            # Provide detailed error analysis
            response_parts.append("\n**Common Issues & Solutions:**")

            # Check for specific error patterns and provide targeted help
            all_error_text = " ".join(str(e) for e in schema_result.get('errors', [])).lower()

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

            if "minitems" in all_error_text or "empty" in all_error_text:
                response_parts.append("  📌 Empty array: 'mode' must have at least one mode defined")

            # Show partial spec for reference
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
    # Extract and fix
    ha_dict, _ = extract_and_fix_ha_spec(ha_spec, auto_fix=True)
    
    if ha_dict is None:
        return False, "Failed to extract specification"
    
    # Validate with JSON Schema
    if not HAS_JSONSCHEMA or validate_ha_with_schema is None:
        return True, "Specification extracted (schema validation unavailable)"
    
    result = validate_ha_with_schema(ha_dict, auto_extract=False)
    
    if result['valid']:
        return True, "Specification is valid"
    else:
        errors = [e['message'] for e in result.get('errors', [])[:3]]
        return False, f"Invalid: {'; '.join(errors)}"


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

