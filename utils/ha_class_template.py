"""
Templates for Hybrid Automaton Python classes with inheritance from base class.

All generated classes inherit from HybridAutomatonBase to ensure:
1. Consistent interface
2. Type checking and validation
3. Common utility methods
"""

from typing import Optional


def get_base_class_import() -> str:
    """Return the import statement for the base class."""
    return "from utils.ha_base_class import HybridAutomatonBase\n"


def get_simple_ha_template(num_variables: int, num_inputs: int, class_name: str = "HybridAutomaton") -> str:
    """Generate minimal HybridAutomaton class template."""
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else ""
    order = 2 if num_variables == 1 else 1

    # Build dynamics example based on order
    if order == 2:
        dynamics_example = 'f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0]"'
    elif num_variables == 1:
        dynamics_example = 'f"x1[1] = {self.params[0]}*x1[0]"'
    else:
        eqs = [f"x{i+1}[1] = {{self.params[{i}]}}*x{i+1}[0]" for i in range(num_variables)]
        dynamics_example = f'f"{", ".join(eqs)}"'

    template = f'''{get_base_class_import()}

class {class_name}(HybridAutomatonBase):
    def __init__(self):
        self.params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.var = "{var_names}"
        self.input = "{input_names}"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = {order}

    def num_modes(self) -> int:
        return 1  # TODO: infer from data

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        if mode_id == 1:
            return {dynamics_example}  # TODO: correct dynamics
        raise ValueError(f"Unknown mode_id: {{mode_id}}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        return False  # TODO: add transitions if multiple modes

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        pass  # TODO: add resets if needed
'''
    return template


def get_example_duffing_template() -> str:
    """
    Example: Pre-filled Duffing oscillator template for testing.

    This shows what a well-specified HA looks like with inheritance.
    """
    return f'''{get_base_class_import()}

class DuffingOscillator(HybridAutomatonBase):
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
            return f"x1[2] = {{self.params[0]}}*x1[1] + {{self.params[1]}}*x1[0] + {{self.params[2]}}*x1[0]**3 + {{self.params[3]}}*u['u1']"
        else:
            raise ValueError(f"Unknown mode_id: {{mode_id}}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        return False  # Single mode, no transitions

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        pass  # No resets needed


# Alias for compatibility
HybridAutomaton = DuffingOscillator
'''


def get_bouncing_ball_template() -> str:
    """
    Example: Bouncing ball - a multi-mode system with resets.

    Demonstrates guard conditions and reset maps.
    """
    return f'''{get_base_class_import()}

class BouncingBall(HybridAutomatonBase):
    """
    Bouncing ball hybrid automaton.

    Mode 1: Free fall (y'' = -g)
    Mode 2: Impact (velocity reset)

    Guard: y <= 0 triggers impact
    Reset: v -> -e*v (coefficient of restitution)
    """

    def __init__(self):
        self.params = [
            -9.81,  # params[0]: gravity
            0.0,    # params[1]: ground level
            0.8,    # params[2]: coefficient of restitution
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # params[3-9]: unused
        ]

        self.var = "x1"  # height
        self.input = ""
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2

    def num_modes(self) -> int:
        return 2  # Free fall and impact

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        if mode_id == 1:
            # Free fall: y'' = g (gravity)
            return f"x1[2] = {{self.params[0]}}"
        elif mode_id == 2:
            # Same dynamics during impact mode
            return f"x1[2] = {{self.params[0]}}"
        else:
            raise ValueError(f"Unknown mode_id: {{mode_id}}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        if source_mode == 1 and target_mode == 2:
            # Transition to impact when height <= ground and falling
            return x['x1'][0] <= self.params[1] and x['x1'][1] < 0
        elif source_mode == 2 and target_mode == 1:
            # Return to free fall after reset
            return True
        return False

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        if source_mode == 1 and target_mode == 2:
            # Velocity reversal with restitution
            x['x1'][1] = -self.params[2] * x['x1'][1]
            # Ensure position is at ground level
            x['x1'][0] = self.params[1]


# Alias for compatibility
HybridAutomaton = BouncingBall
'''


def get_two_tank_template() -> str:
    """
    Example: Two-tank system - coupled first-order system.

    Demonstrates multi-variable systems.
    """
    return f'''{get_base_class_import()}

class TwoTankSystem(HybridAutomatonBase):
    """
    Two-tank hydraulic system.

    x1: Level in tank 1
    x2: Level in tank 2

    Dynamics:
    x1' = (u1 - k1*x1) / A1
    x2' = (k1*x1 - k2*x2) / A2
    """

    def __init__(self):
        self.params = [
            0.1,   # params[0]: k1/A1 - flow coefficient tank 1
            -0.1,  # params[1]: -k1/A1
            0.1,   # params[2]: k1/A2 - inflow to tank 2
            -0.05, # params[3]: -k2/A2 - outflow from tank 2
            1.0,   # params[4]: input gain
            0.0, 0.0, 0.0, 0.0, 0.0  # params[5-9]: unused
        ]

        self.var = "x1, x2"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 1

    def num_modes(self) -> int:
        return 1

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        if mode_id == 1:
            # Coupled tank dynamics
            return (
                f"x1[1] = {{self.params[0]}}*u['u1'] + {{self.params[1]}}*x1[0], "
                f"x2[1] = {{self.params[2]}}*x1[0] + {{self.params[3]}}*x2[0]"
            )
        else:
            raise ValueError(f"Unknown mode_id: {{mode_id}}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        return False  # Single mode

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        pass


# Alias for compatibility
HybridAutomaton = TwoTankSystem
'''


def generate_class_for_system(
    system_name: str,
    num_variables: int,
    num_inputs: int,
    dynamics_template: Optional[str] = None
) -> str:
    """
    Generate a named HA class for a specific system.

    Args:
        system_name: Name of the system (e.g., "duffing", "ball")
        num_variables: Number of state variables
        num_inputs: Number of input variables
        dynamics_template: Optional pre-filled dynamics string

    Returns:
        Complete Python class code
    """
    # Convert system_name to CamelCase class name
    class_name = ''.join(word.capitalize() for word in system_name.split('_'))
    class_name += "HA"  # e.g., "DuffingHA", "BouncingBallHA"

    return get_simple_ha_template(num_variables, num_inputs, class_name=class_name)


# Quick test
if __name__ == "__main__":
    print("=== Single Variable, 2nd Order (Oscillator) ===")
    print(get_simple_ha_template(num_variables=1, num_inputs=1))

    print("\n" + "="*80 + "\n")
    print("=== Two Variables, 1st Order ===")
    print(get_simple_ha_template(num_variables=2, num_inputs=1))

    print("\n" + "="*80 + "\n")
    print("=== Duffing Example ===")
    print(get_example_duffing_template())

    print("\n" + "="*80 + "\n")
    print("=== Bouncing Ball Example ===")
    print(get_bouncing_ball_template())

    print("\n" + "="*80 + "\n")
    print("=== Named Class for 'duffing' ===")
    print(generate_class_for_system("duffing", 1, 1))
