import os
import sys

# Import JSON Schema for prompt generation
from .ha_json_schema import HA_JSON_SCHEMA, get_schema_for_prompt


# Valid JSON template for HA specification (parsed by json.loads)
# Example: one mode with reset
one_mode_reset_ha_spec_prompt = """{
    "automaton": {
        "var": "x1, x2",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x2[0], x2[1] = x1[0] + u1"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 1",
                "condition": "x1 <= 0",
                "reset": {
                    "x1": [0],
                    "x2": ["-0.1 * x1[0]"]
                }
            }
        ]
    },
    "config": {
        "dt": 0.0001,
        "total_time": 2.0,
        "order": 1,
        "need_reset": true,
        "non_linear_items": ""
    }
}"""

# Example: Single-mode system (2nd-order ODE, 1 variable)
second_order_ha_spec_prompt = """{
    "automaton": {
        "var": "x1",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[2] = x1[1] + x1[0] + x1[0] ** 2 + u1"
            }
        ],
        "edge": []
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "order": 2,
        "need_reset": false,
        "non_linear_items": "x1[?] ** 2"
    }
}"""

# Example: Two-mode system
two_modes_ha_spec_prompt = """{
    "automaton": {
        "var": "x1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x1[0] + 5"
            },
            {
                "id": 2,
                "eq": "x1[1] = -x1[0] - 3"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 2",
                "condition": "x1 >= 4"
            },
            {
                "direction": "2 -> 1",
                "condition": "x1 <= 9"
            }
        ]
    },
    "config": {
        "dt": 0.01,
        "total_time": 20.0,
        "order": 1,
        "need_reset": false,
        "non_linear_items": ""
    }
}"""


# JSON Schema-based documentation for structured output
def get_ha_spec_documentation_with_schema(simplified: bool = True) -> str:
    """
    Generate HA specification documentation that includes the JSON Schema.

    This provides a machine-readable specification that LLMs can follow more precisely.

    Args:
        simplified: If True, use simplified schema without $ref definitions

    Returns:
        Documentation string with embedded JSON Schema
    """
    schema_str = get_schema_for_prompt(include_definitions=not simplified)

    return f"""## Hybrid Automaton Specification Format (JSON Schema)

### JSON Schema Definition
Your output MUST conform to this JSON Schema:

```json
{schema_str}
```


### Examples

**Single-mode 2nd-order:**
```json
{second_order_ha_spec_prompt}
```

**Two-mode switching:**
```json
{two_modes_ha_spec_prompt}
```

**With reset map:**
```json
{one_mode_reset_ha_spec_prompt}
```
"""

ha_spec_docs = get_ha_spec_documentation_with_schema()

if __name__ == "__main__":


    print("\n" + "=" * 60)
    print("JSON SCHEMA-BASED DOCUMENTATION")
    print("=" * 60)
    print(ha_spec_docs)
    # # save the documentation to a file (use absolute path based on script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "ha_spec_docs.md")
    with open(output_path, "w") as f:
        f.write(ha_spec_docs)
    print(f"\nDocumentation saved to: {output_path}")



