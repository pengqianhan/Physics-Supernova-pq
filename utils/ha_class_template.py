"""
Simple initial templates for Hybrid Automaton Python classes.

Inspired by SR-Scientist's skeleton approach - provide LLM with a minimal
starting point that it can iteratively improve.
"""


def get_simple_ha_template(num_variables: int, num_inputs: int) -> str:
    """
    Generate a simple, minimal HybridAutomaton class template.

    This serves as the starting point (v0) for LLM-driven iterative refinement.

    Args:
        num_variables: Number of state variables
        num_inputs: Number of input variables

    Returns:
        Python class code string
    """
    # Generate variable and input names
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else ""

    # Determine order based on common patterns
    # Single variable systems are often 2nd order (oscillators)
    # Multi-variable systems are often 1st order
    order = 2 if num_variables == 1 else 1

    template = f'''class HybridAutomaton:
    """
    Hybrid Automaton specification.

    TODO: Infer the correct dynamics, modes, and switching conditions from data.
    """

    def __init__(self):
        # Numerical parameters (tune these to fit the data)
        self.params = [
            0.0,  # params[0]: placeholder
            0.0,  # params[1]: placeholder
            0.0,  # params[2]: placeholder
            0.0,  # params[3]: placeholder
            0.0,  # params[4]: placeholder
            0.0, 0.0, 0.0, 0.0, 0.0  # params[5-9]: unused
        ]

        # System structure (FIXED - do not change these)
        self.var = "{var_names}"
        self.input = "{input_names}"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = {order}

    def num_modes(self) -> int:
        """Return number of discrete modes."""
        # TODO: Determine correct number of modes from data
        return 1

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        """
        Return ODE equation string for the given mode.

        Args:
            mode_id: Mode identifier (1-indexed)
            x: State dict {{var_name: [x[0], x[1], ...]}}
            u: Input dict {{input_name: value}}

        Returns:
            ODE equation string (e.g., "x1[2] = params[0]*x1[1] + params[1]*x1[0] + u1")
        """
        if mode_id == 1:
            # TODO: Replace with correct dynamics
            # For order={order} system, left side should be highest derivative
'''

    # Add mode-specific dynamics template
    if order == 1:
        if num_variables == 1:
            template += f'''            return f"x1[1] = {{self.params[0]}}*x1[0]"  # Simple linear dynamics
'''
        else:
            # Multi-variable 1st order
            equations = []
            for i in range(num_variables):
                equations.append(f"x{i+1}[1] = {{self.params[{i}]}}*x{i+1}[0]")
            eq_str = ", ".join(equations)
            template += f'''            return f"{eq_str}"  # Coupled dynamics
'''
    else:  # order == 2
        template += f'''            # Example 2nd-order dynamics
            return f"x1[2] = {{self.params[0]}}*x1[1] + {{self.params[1]}}*x1[0]"
'''

    # Add rest of methods
    template += '''        else:
            raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        """
        Evaluate guard condition for mode transition.

        Returns True if transition should occur.
        """
        # TODO: Add switching conditions if multiple modes exist
        # Example: if source_mode == 1 and target_mode == 2:
        #              return x['x1'][0] >= self.params[3]
        return False

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        """
        Apply state reset on transition (modifies x in-place).

        Leave empty if no resets needed.
        """
        # TODO: Add reset logic if needed
        # Example: x['x1'][0] = -0.9 * x['x1'][0]
        pass
'''

    return template


def get_example_duffing_template() -> str:
    """
    Example: Pre-filled Duffing oscillator template for testing.

    This shows what a well-specified HA looks like.
    """
    return '''class HybridAutomaton:
    """Duffing oscillator: x'' + damping*x' + k1*x + k3*x^3 = u"""

    def __init__(self):
        self.params = [
            -0.15,  # params[0]: damping coefficient
            -1.0,   # params[1]: linear stiffness
            -1.0,   # params[2]: cubic stiffness
            1.0,    # params[3]: input gain
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # params[4-9]: unused
        ]

        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2

    def num_modes(self) -> int:
        return 1

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        if mode_id == 1:
            # Duffing equation: x1[2] = damping*x1[1] + k1*x1[0] + k3*x1[0]^3 + gain*u1
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*x1[0]**3 + {self.params[3]}*u['u1']"
        else:
            raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        return False  # Single mode, no transitions

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        pass  # No resets needed
'''


# Quick test
if __name__ == "__main__":
    print("=== Single Variable, 2nd Order (Oscillator) ===")
    print(get_simple_ha_template(num_variables=1, num_inputs=1))

    print("\n=== Two Variables, 1st Order ===")
    print(get_simple_ha_template(num_variables=2, num_inputs=1))

    print("\n=== Duffing Example ===")
    print(get_example_duffing_template())
