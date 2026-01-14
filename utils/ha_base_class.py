"""
Abstract Base Class for Hybrid Automaton Python Class Generation.

This module provides the base class that all generated Hybrid Automaton
classes should inherit from. Using inheritance ensures:
1. Consistent interface across all HA implementations
2. Type checking and validation
3. Common utility methods and default implementations
4. Clear contract for what methods must be implemented
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class HybridAutomatonBase(ABC):
    """
    Abstract base class for Hybrid Automaton specifications.

    All generated HA classes should inherit from this base class and implement
    the abstract methods. The base class provides:
    - Required attribute declarations
    - Default implementations where applicable
    - Validation utilities
    - Common helper methods

    Example Usage:
    ```python
    class DuffingOscillator(HybridAutomatonBase):
        def __init__(self):
            self.params = [-0.15, -1.0, -1.0, 1.0, 0, 0, 0, 0, 0, 0]
            self.var = "x1"
            self.input = "u1"
            self.dt = 0.001
            self.total_time = 10.0
            self.order = 2

        def num_modes(self) -> int:
            return 1

        def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*x1[0]**3 + u['u1']"

        def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
            return False

        def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
            pass
    ```
    """

    # Required attributes (subclasses must set these in __init__)
    params: List[float]      # Tunable parameters (max 10 elements)
    var: str                 # State variable names, comma-separated (e.g., "x1, x2")
    input: str               # Input variable names, comma-separated (e.g., "u1") or ""
    dt: float                # Integration time step (default: 0.001)
    total_time: float        # Total simulation duration (default: 10.0)
    order: int               # ODE order (1 or 2)

    @abstractmethod
    def num_modes(self) -> int:
        """
        Return the number of discrete modes in the hybrid automaton.

        Returns:
            int: Number of modes (must be >= 1)
        """
        pass

    @abstractmethod
    def mode_dynamics(self, mode_id: int, x: Dict[str, List[float]], u: Dict[str, float]) -> str:
        """
        Return the ODE equation string for the given mode.

        The equation string defines the dynamics for each state variable
        in the specified mode. The format follows the convention:
        - `x[0]` = variable value
        - `x[1]` = first derivative (dx/dt)
        - `x[2]` = second derivative (d²x/dt²)
        - Left side must be the highest derivative

        Args:
            mode_id: Mode identifier (1-indexed, range: 1 to num_modes())
            x: State dictionary mapping variable names to their value/derivative list
               e.g., {'x1': [position, velocity], 'x2': [value, derivative]}
            u: Input dictionary mapping input names to current values
               e.g., {'u1': 0.5, 'u2': 1.0}

        Returns:
            str: ODE equation string, e.g., "x1[2] = -0.5*x1[1] - 5*x1[0] + u['u1']"

        Example:
            For a damped oscillator: x'' + 0.5x' + 5x = u
            ```python
            def mode_dynamics(self, mode_id, x, u):
                if mode_id == 1:
                    return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + u['u1']"
            ```
        """
        pass

    @abstractmethod
    def guard_condition(
        self,
        source_mode: int,
        target_mode: int,
        x: Dict[str, List[float]],
        u: Dict[str, float]
    ) -> bool:
        """
        Evaluate the guard condition for a mode transition.

        This method is called to check if the system should transition
        from source_mode to target_mode given the current state.

        Args:
            source_mode: Current mode ID (1-indexed)
            target_mode: Potential next mode ID (1-indexed)
            x: Current state dictionary (same format as mode_dynamics)
            u: Current input dictionary

        Returns:
            bool: True if the transition should occur, False otherwise

        Example:
            ```python
            def guard_condition(self, source_mode, target_mode, x, u):
                if source_mode == 1 and target_mode == 2:
                    return x['x1'][0] >= self.params[3]  # Switch when x1 >= threshold
                elif source_mode == 2 and target_mode == 1:
                    return x['x1'][0] < self.params[4]   # Switch back
                return False
            ```
        """
        pass

    @abstractmethod
    def reset_map(
        self,
        source_mode: int,
        target_mode: int,
        x: Dict[str, List[float]],
        u: Dict[str, float]
    ) -> None:
        """
        Apply state reset on mode transition (modifies x in-place).

        This method is called when a transition occurs to apply any
        state resets (e.g., velocity reversal on impact).

        Args:
            source_mode: Mode being left (1-indexed)
            target_mode: Mode being entered (1-indexed)
            x: State dictionary to modify in-place
            u: Current input dictionary

        Example:
            ```python
            def reset_map(self, source_mode, target_mode, x, u):
                if source_mode == 1 and target_mode == 2:
                    # Reverse velocity with coefficient of restitution
                    x['x1'][1] = -self.params[5] * x['x1'][1]
            ```
        """
        pass

    # =====================================================================
    # Utility Methods (non-abstract, inherited by all subclasses)
    # =====================================================================

    def get_var_names(self) -> List[str]:
        """Parse and return list of state variable names."""
        return [v.strip() for v in self.var.split(',') if v.strip()]

    def get_input_names(self) -> List[str]:
        """Parse and return list of input variable names."""
        if not self.input:
            return []
        return [i.strip() for i in self.input.split(',') if i.strip()]

    def get_num_vars(self) -> int:
        """Return number of state variables."""
        return len(self.get_var_names())

    def get_num_inputs(self) -> int:
        """Return number of input variables."""
        return len(self.get_input_names())

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the HA specification for common errors.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required attributes
        if not hasattr(self, 'params') or not isinstance(self.params, list):
            errors.append("'params' must be a list")
        elif len(self.params) > 10:
            errors.append("'params' must have at most 10 elements")

        if not hasattr(self, 'var') or not self.var:
            errors.append("'var' must be a non-empty string")

        if not hasattr(self, 'dt') or self.dt <= 0:
            errors.append("'dt' must be a positive number")

        if not hasattr(self, 'total_time') or self.total_time <= 0:
            errors.append("'total_time' must be a positive number")

        if not hasattr(self, 'order') or self.order not in (1, 2):
            errors.append("'order' must be 1 or 2")

        # Check num_modes
        try:
            n_modes = self.num_modes()
            if not isinstance(n_modes, int) or n_modes < 1:
                errors.append("num_modes() must return a positive integer")
        except Exception as e:
            errors.append(f"num_modes() raised exception: {e}")

        return len(errors) == 0, errors

    def get_config(self) -> Dict:
        """Return configuration dictionary for serialization."""
        return {
            'var': self.var,
            'input': getattr(self, 'input', ''),
            'dt': getattr(self, 'dt', 0.001),
            'total_time': getattr(self, 'total_time', 10.0),
            'order': getattr(self, 'order', 1),
            'num_modes': self.num_modes(),
            'num_params': len(getattr(self, 'params', [])),
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"var='{self.var}', input='{getattr(self, 'input', '')}', "
            f"modes={self.num_modes()}, order={getattr(self, 'order', 1)})"
        )

    def to_json(self) -> Dict:
        """
        Convert the HA specification to JSON format for legacy simulator.

        This provides compatibility with HybridAutomata.from_json().

        Returns:
            Dictionary in JSON format with 'automaton' and 'config' sections
        """
        # Build mode equations
        modes = []
        for mode_id in range(1, self.num_modes() + 1):
            # Create dummy state and input dicts for equation generation
            x_dummy = {var: [0.0] * getattr(self, 'order', 1)
                       for var in self.get_var_names()}
            u_dummy = {inp: 0.0 for inp in self.get_input_names()}

            eq_str = self.mode_dynamics(mode_id, x_dummy, u_dummy)
            # Convert u['u1'] notation to u1 for legacy format
            eq_str = eq_str.replace("u['", "").replace("']", "")
            modes.append({"id": mode_id, "eq": eq_str})

        # Build edges (guards)
        edges = []
        for src in range(1, self.num_modes() + 1):
            for tgt in range(1, self.num_modes() + 1):
                if src != tgt:
                    # Check if there's a guard condition defined
                    # We can't easily extract guard conditions as strings,
                    # so leave edges empty for now (guards handled dynamically)
                    pass

        return {
            "automaton": {
                "var": self.var,
                "input": getattr(self, 'input', ''),
                "mode": modes,
                "edge": edges
            },
            "config": {
                "dt": getattr(self, 'dt', 0.001),
                "total_time": getattr(self, 'total_time', 10.0),
                "order": getattr(self, 'order', 1),
                "need_reset": False,
                "non_linear_items": ""
            }
        }


# Alias for backwards compatibility and convenience
HybridAutomaton = HybridAutomatonBase
