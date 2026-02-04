"""
Hybrid Automaton Parameter Optimizer

This module provides automated optimization of numeric parameters in Hybrid Automaton
specifications using Nevergrad gradient-free optimization. It extracts all numeric
values from ODEs, guard conditions, and reset expressions, then optimizes them to
minimize trajectory error against ground truth data.

Key Features:
- Automatic parameter extraction from HA specifications using regex
- Configurable bounds heuristics based on parameter context
- Comparison of original vs optimized specifications
- Supports arbitrary HA structures (multiple modes, edges, resets)

Example:
    from utils.ha_param_optimizer import optimize_ha_parameters

    result = optimize_ha_parameters(
        ha_input=ha_spec,
        npz_file_path='data_all/ATVA/ball_g/ground_truth_0.npz',
        budget=100
    )
    better_ha = result.get_better_spec()
"""

import re
import json
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional, Callable
import os
import sys

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dainarx_dir = os.path.join(_current_dir, 'Dainarx_code')
if _dainarx_dir not in sys.path:
    sys.path.insert(0, _dainarx_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from HA_evaluation import HAEvaluator

# Import ha_spec_validator for preprocessing (handles dict extraction from text)
from ha_spec_validator import extract_and_fix_ha_spec


# ============================================================================
# Constants
# ============================================================================

# Regex pattern to extract numeric values from expressions
# Matches: integers, decimals, scientific notation
# Negative lookbehind: avoid matching array indices like [0], [1]
# Negative lookbehind: avoid matching variable names like x1, u2
NUMERIC_PATTERN = r'(?<![a-zA-Z_\[])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?!\])'

# Alternative pattern that's more conservative (no negative numbers captured separately)
POSITIVE_NUMERIC_PATTERN = r'(?<![a-zA-Z_\[\d])(\d+\.?\d*(?:[eE][+-]?\d+)?)'


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NumericParam:
    """
    Represents a single extractable numeric parameter from an HA specification.

    Attributes:
        index: Global parameter index (params[index] in optimization)
        value: Original numeric value
        location: Where the parameter comes from ('mode_eq', 'edge_condition', 'edge_reset')
        path: Path to the containing field, e.g., ('mode', 0, 'eq') or ('edge', 1, 'reset', 'x2', 0)
        position: (start, end) character positions in the original string
        bounds: (lower, upper) bounds for optimization
        original_string: The original string containing this parameter
    """
    index: int
    value: float
    location: str
    path: tuple
    position: tuple  # (start, end) in string
    bounds: tuple  # (lower, upper)
    original_string: str = ""


@dataclass
class OptimizationResult:
    """
    Result from HA parameter optimization.

    Attributes:
        original_ha: Original HA specification
        optimized_ha: HA specification with optimized parameters
        original_error: Mean trajectory difference with original parameters
        optimized_error: Mean trajectory difference with optimized parameters
        parameters: List of NumericParam objects that were optimized
        original_values: Array of original parameter values
        optimized_values: Array of optimized parameter values
        num_evaluations: Number of objective function evaluations
        improvement_pct: Percentage improvement (positive means optimized is better)
    """
    original_ha: Dict[str, Any]
    optimized_ha: Dict[str, Any]
    original_error: float
    optimized_error: float
    parameters: List[NumericParam]
    original_values: np.ndarray
    optimized_values: np.ndarray
    num_evaluations: int = 0
    improvement_pct: float = 0.0

    def __post_init__(self):
        """Calculate improvement percentage after initialization."""
        if self.original_error > 0:
            self.improvement_pct = ((self.original_error - self.optimized_error) / self.original_error) * 100

    def get_better_spec(self) -> Dict[str, Any]:
        """Return the specification with lower error."""
        if self.optimized_error < self.original_error:
            return self.optimized_ha
        return self.original_ha

    def is_improved(self) -> bool:
        """Check if optimization improved the result."""
        return self.optimized_error < self.original_error

    def summary(self) -> str:
        """Generate a formatted summary of the optimization results."""
        lines = [
            "=" * 80,
            "HA Parameter Optimization Results",
            "=" * 80,
            f"Original Error:  {self.original_error:.6f}",
            f"Optimized Error: {self.optimized_error:.6f}",
            f"Improvement:     {self.improvement_pct:.2f}%",
            f"Evaluations:     {self.num_evaluations}",
            "",
            "Parameter Changes:",
            f"  {'#':<3} | {'Location':<20} | {'Original':<12} | {'Optimized':<12} | {'Bounds'}",
            "-" * 80,
        ]

        for i, param in enumerate(self.parameters):
            orig_val = self.original_values[i]
            opt_val = self.optimized_values[i]
            bounds_str = f"[{param.bounds[0]:.3g}, {param.bounds[1]:.3g}]"
            path_str = self._format_path(param.path)
            lines.append(f"  {i:<3} | {path_str:<20} | {orig_val:<12.6f} | {opt_val:<12.6f} | {bounds_str}")

        lines.append("")
        better = "OPTIMIZED" if self.is_improved() else "ORIGINAL"
        lines.append(f"Better Specification: {better}")
        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def _format_path(path: tuple) -> str:
        """Format a path tuple for display."""
        parts = []
        for p in path:
            if isinstance(p, int):
                parts.append(f"[{p}]")
            else:
                if parts:
                    parts.append(f".{p}")
                else:
                    parts.append(str(p))
        return "".join(parts)


# ============================================================================
# ParameterizedHA Class
# ============================================================================

class ParameterizedHA:
    """
    Manages parameter extraction and reconstruction for Hybrid Automaton specifications.

    This class handles:
    - Extracting all numeric parameters from ODE equations, guard conditions, and resets
    - Computing appropriate bounds for each parameter
    - Reconstructing an HA specification from a parameter array
    """

    def __init__(self, ha_dict: Dict[str, Any], parameter_filter: Optional[Callable[[NumericParam], bool]] = None):
        """
        Initialize with an HA specification.

        Args:
            ha_dict: HA specification dictionary with 'automaton' and 'config' keys
            parameter_filter: Optional function to filter which parameters to extract
        """
        self.original_ha = copy.deepcopy(ha_dict)
        self.parameter_filter = parameter_filter
        self.parameters: List[NumericParam] = []
        self._extract_parameters()

    def _compute_bounds(self, value: float, location: str, context: str = "") -> Tuple[float, float]:
        """
        Compute optimization bounds for a parameter value based on heuristics.

        The bounds are computed to:
        1. Always include the original value
        2. Provide reasonable exploration range based on the value's magnitude
        3. Handle negative values correctly (bounds may span negative range)

        Args:
            value: The numeric value
            location: Where the parameter comes from ('mode_eq', 'edge_condition', 'edge_reset')
            context: Additional context string for the parameter

        Returns:
            Tuple of (lower_bound, upper_bound) where lower < upper
        """
        abs_val = abs(value)

        # Handle zero specially - allow small positive and negative values
        if abs_val < 1e-10:
            return (-1.0, 1.0)

        # Handle coefficients in range (0, 1) - likely damping, restitution, etc.
        if 0 < abs_val < 1:
            if value > 0:
                # Positive coefficient (e.g., damping)
                if location == 'edge_reset':
                    return (0.01, 0.99)
                return (0.01, 2.0)
            else:
                # Negative coefficient (e.g., -0.9 restitution multiplier)
                if location == 'edge_reset':
                    return (-0.99, -0.01)
                return (-2.0, -0.01)

        # Handle coefficients around 1
        if 0.5 <= abs_val <= 2.0:
            if value > 0:
                return (0.1, 5.0)
            else:
                return (-5.0, -0.1)

        # Handle gravity-like constants (5-15 range)
        if 5.0 <= abs_val <= 15.0:
            if value > 0:
                return (1.0, 30.0)
            else:
                # Negative gravity (e.g., -9.8)
                return (-30.0, -1.0)

        # Handle threshold values in conditions
        if location == 'edge_condition':
            # For thresholds, allow 0.5x to 2x scaling, preserving sign
            margin = abs_val * 0.75  # Allow 75% variation
            return (value - margin, value + margin)

        # General heuristic: scale by 0.1x to 10x, preserving sign
        if value > 0:
            return (max(0.001, value * 0.1), value * 10.0)
        else:
            return (value * 10.0, min(-0.001, value * 0.1))

    def _extract_numbers_from_string(self, s: str, location: str, path: tuple) -> List[NumericParam]:
        """
        Extract numeric values from a string expression.

        Args:
            s: The string expression (equation, condition, or reset)
            location: Parameter location type
            path: Path to this string in the HA structure

        Returns:
            List of NumericParam objects
        """
        params = []

        # Find all numeric matches
        for match in re.finditer(POSITIVE_NUMERIC_PATTERN, s):
            value_str = match.group(1)
            start, end = match.start(1), match.end(1)

            # Skip if this looks like an array index [0], [1], etc.
            if start > 0 and s[start-1] == '[':
                continue

            # Check if preceded by minus sign (not subtraction operator)
            is_negative = False
            if start > 0:
                # Look backwards to find if there's a minus
                before = s[:start].rstrip()
                if before and before[-1] == '-':
                    # Check if it's unary minus (not subtraction)
                    # Unary if at start, after '=', after ',' or after '('
                    before_stripped = before[:-1].rstrip()
                    if not before_stripped or before_stripped[-1] in '=,(':
                        is_negative = True
                        start = len(before) - 1  # Include the minus sign

            try:
                value = float(value_str)
                if is_negative:
                    value = -value
            except ValueError:
                continue

            # Skip very small values that are likely noise
            if abs(value) < 1e-15:
                continue

            bounds = self._compute_bounds(value, location)

            param = NumericParam(
                index=len(self.parameters) + len(params),
                value=value,
                location=location,
                path=path,
                position=(start, end),
                bounds=bounds,
                original_string=s
            )

            # Apply filter if provided
            if self.parameter_filter is None or self.parameter_filter(param):
                params.append(param)

        return params

    def _extract_parameters(self):
        """Extract all numeric parameters from the HA specification."""
        self.parameters = []
        automaton = self.original_ha.get('automaton', {})

        # Extract from mode equations
        for i, mode in enumerate(automaton.get('mode', [])):
            if 'eq' in mode and isinstance(mode['eq'], str):
                params = self._extract_numbers_from_string(
                    mode['eq'],
                    'mode_eq',
                    ('mode', i, 'eq')
                )
                self.parameters.extend(params)

        # Extract from edges
        for i, edge in enumerate(automaton.get('edge', [])):
            # Extract from conditions
            if 'condition' in edge and isinstance(edge['condition'], str):
                params = self._extract_numbers_from_string(
                    edge['condition'],
                    'edge_condition',
                    ('edge', i, 'condition')
                )
                self.parameters.extend(params)

            # Extract from resets
            if 'reset' in edge and isinstance(edge['reset'], dict):
                for var_name, reset_exprs in edge['reset'].items():
                    if isinstance(reset_exprs, list):
                        for j, expr in enumerate(reset_exprs):
                            if isinstance(expr, str):
                                params = self._extract_numbers_from_string(
                                    expr,
                                    'edge_reset',
                                    ('edge', i, 'reset', var_name, j)
                                )
                                self.parameters.extend(params)
                            elif isinstance(expr, (int, float)):
                                # Direct numeric value in reset
                                value = float(expr)

                                # Skip simple boundary constants (0, 1) when they appear
                                # alone as direct reset values - these are usually fixed boundaries
                                if abs(value) < 1e-10 or abs(value - 1.0) < 1e-10:
                                    continue

                                bounds = self._compute_bounds(value, 'edge_reset')
                                param = NumericParam(
                                    index=len(self.parameters),
                                    value=value,
                                    location='edge_reset',
                                    path=('edge', i, 'reset', var_name, j),
                                    position=(0, 0),  # Not a string, no position
                                    bounds=bounds,
                                    original_string=str(expr)
                                )
                                if self.parameter_filter is None or self.parameter_filter(param):
                                    self.parameters.append(param)

        # Re-index parameters
        for i, param in enumerate(self.parameters):
            param.index = i

    def get_parameter_count(self) -> int:
        """Return the number of extractable parameters."""
        return len(self.parameters)

    def get_original_values(self) -> np.ndarray:
        """Return array of original parameter values."""
        return np.array([p.value for p in self.parameters])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds as arrays."""
        if not self.parameters:
            return np.array([]), np.array([])
        lower = np.array([p.bounds[0] for p in self.parameters])
        upper = np.array([p.bounds[1] for p in self.parameters])
        return lower, upper

    def reconstruct(self, params: np.ndarray) -> Dict[str, Any]:
        """
        Reconstruct an HA specification with new parameter values.

        Args:
            params: Array of parameter values (must match parameter count)

        Returns:
            New HA specification dictionary
        """
        if len(params) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} parameters, got {len(params)}")

        # Deep copy the original
        result = copy.deepcopy(self.original_ha)
        automaton = result.get('automaton', {})

        # Group parameters by their containing string
        # This is important because we need to replace multiple values in the same string
        string_replacements: Dict[tuple, List[Tuple[NumericParam, float]]] = {}

        for param, new_value in zip(self.parameters, params):
            # Handle direct numeric values in reset (not string expressions)
            if param.position == (0, 0) and param.location == 'edge_reset':
                # This is a direct numeric value, handle separately
                path = param.path
                if path[0] == 'edge':
                    edge_idx = path[1]
                    var_name = path[3]
                    expr_idx = path[4]
                    edge = automaton['edge'][edge_idx]
                    if isinstance(edge['reset'][var_name][expr_idx], (int, float)):
                        edge['reset'][var_name][expr_idx] = new_value
                continue

            # Group string replacements by path
            key = param.path
            if key not in string_replacements:
                string_replacements[key] = []
            string_replacements[key].append((param, new_value))

        # Apply string replacements (in reverse position order to preserve indices)
        for path, replacements in string_replacements.items():
            # Sort by position in reverse order
            replacements.sort(key=lambda x: x[0].position[0], reverse=True)

            # Get the original string
            obj = automaton
            for p in path[:-1]:
                if isinstance(p, int):
                    obj = obj[p]
                else:
                    obj = obj[p]

            original_string = obj[path[-1]]
            new_string = original_string

            # Apply replacements from end to start
            for param, new_value in replacements:
                start, end = param.position
                # Format the new value
                if abs(new_value) < 0.001 or abs(new_value) >= 1000:
                    new_value_str = f"{new_value:.6e}"
                else:
                    new_value_str = f"{new_value:.6f}".rstrip('0').rstrip('.')

                new_string = new_string[:start] + new_value_str + new_string[end:]

            # Set the new string
            obj[path[-1]] = new_string

        return result


# ============================================================================
# HAParamOptimizer Class
# ============================================================================

class HAParamOptimizer:
    """
    Main optimizer class that uses Nevergrad to optimize HA parameters.
    """

    def __init__(
        self,
        ha_dict: Dict[str, Any],
        npz_file_path: str,
        dt: Optional[float] = None,
        total_time: Optional[float] = None,
        parameter_filter: Optional[Callable[[NumericParam], bool]] = None
    ):
        """
        Initialize the optimizer.

        Args:
            ha_dict: HA specification dictionary
            npz_file_path: Path to ground truth NPZ file
            dt: Time step (uses config value if not specified)
            total_time: Total simulation time (uses config value if not specified)
            parameter_filter: Optional function to filter which parameters to optimize
        """
        self.original_ha = copy.deepcopy(ha_dict)
        self.npz_file_path = npz_file_path

        # Extract timing from config
        config = ha_dict.get('config', {})
        self.dt = dt if dt is not None else config.get('dt', 0.001)
        self.total_time = total_time if total_time is not None else config.get('total_time', 10.0)

        # Create parametrized HA
        self.parametrized_ha = ParameterizedHA(ha_dict, parameter_filter)

        # Tracking
        self.eval_count = 0

    def objective(self, params: np.ndarray, verbose: bool = False) -> float:
        """
        Objective function for optimization.

        Args:
            params: Array of parameter values
            verbose: Whether to print intermediate results

        Returns:
            Mean trajectory difference (lower is better)
        """
        self.eval_count += 1

        try:
            # Reconstruct HA with new parameters
            ha_dict = self.parametrized_ha.reconstruct(params)

            # Create evaluator
            evaluator = HAEvaluator(
                ha_dict=ha_dict,
                npz_file_path=self.npz_file_path,
                dt=self.dt,
                total_time=self.total_time
            )

            # Run simulation and compute metrics
            evaluator.load_ground_truth()
            evaluator.simulate()
            metrics = evaluator.compute_metrics()

            # Use mean_diff as the objective
            mean_diff = metrics.get('mean_diff', float('inf'))

            if mean_diff is None:
                mean_diff = float('inf')

            if verbose:
                print(f"  Eval {self.eval_count}: mean_diff={mean_diff:.6f}")

            return mean_diff

        except Exception as e:
            if verbose:
                print(f"  Eval {self.eval_count}: ERROR - {e}")
            return float('inf')

    def optimize(
        self,
        budget: int = 100,
        verbose: bool = True,
        optimizer_class: Optional[str] = None
    ) -> OptimizationResult:
        """
        Run the optimization.

        Args:
            budget: Number of function evaluations
            verbose: Whether to print progress
            optimizer_class: Nevergrad optimizer class name (default: NGOpt)

        Returns:
            OptimizationResult with comparison between original and optimized
        """
        import nevergrad as ng

        # Check if there are parameters to optimize
        num_params = self.parametrized_ha.get_parameter_count()

        if num_params == 0:
            if verbose:
                print("No extractable parameters found. Returning original specification.")

            # Evaluate original
            original_values = np.array([])
            original_error = self.objective(original_values, verbose=False)

            return OptimizationResult(
                original_ha=self.original_ha,
                optimized_ha=self.original_ha,
                original_error=original_error,
                optimized_error=original_error,
                parameters=[],
                original_values=original_values,
                optimized_values=original_values,
                num_evaluations=1
            )

        # Get original values and evaluate
        original_values = self.parametrized_ha.get_original_values()
        self.eval_count = 0
        original_error = self.objective(original_values, verbose=False)

        if verbose:
            print(f"Starting optimization with {num_params} parameters")
            print(f"Original error: {original_error:.6f}")
            print(f"Budget: {budget} evaluations")
            print("-" * 60)

        # Build Nevergrad parametrization
        lower, upper = self.parametrized_ha.get_bounds()

        param_dict = {}
        for i, param in enumerate(self.parametrized_ha.parameters):
            param_dict[f'p{i}'] = ng.p.Scalar(
                init=param.value,
                lower=param.bounds[0],
                upper=param.bounds[1]
            )

        parametrization = ng.p.Instrumentation(**param_dict)

        # Create optimizer
        if optimizer_class is None:
            optimizer = ng.optimizers.NGOpt(
                parametrization=parametrization,
                budget=budget,
                num_workers=1
            )
        else:
            opt_cls = getattr(ng.optimizers, optimizer_class)
            optimizer = opt_cls(
                parametrization=parametrization,
                budget=budget,
                num_workers=1
            )

        # Define objective wrapper
        def nevergrad_objective(**kwargs):
            params = np.array([kwargs[f'p{i}'] for i in range(num_params)])
            return self.objective(params, verbose=verbose)

        # Run optimization
        self.eval_count = 1  # Already evaluated original
        recommendation = optimizer.minimize(nevergrad_objective)

        # Extract optimized values
        optimized_values = np.array([recommendation.kwargs[f'p{i}'] for i in range(num_params)])
        optimized_error = recommendation.loss

        # Reconstruct optimized HA
        optimized_ha = self.parametrized_ha.reconstruct(optimized_values)

        if verbose:
            print("-" * 60)
            print(f"Optimization complete!")
            print(f"Optimized error: {optimized_error:.6f}")
            improvement = ((original_error - optimized_error) / original_error * 100) if original_error > 0 else 0
            print(f"Improvement: {improvement:.2f}%")

        return OptimizationResult(
            original_ha=self.original_ha,
            optimized_ha=optimized_ha,
            original_error=original_error,
            optimized_error=optimized_error,
            parameters=self.parametrized_ha.parameters,
            original_values=original_values,
            optimized_values=optimized_values,
            num_evaluations=self.eval_count
        )


# ============================================================================
# Main Function
# ============================================================================

def optimize_ha_parameters(
    ha_input: Any,
    npz_file_path: str,
    budget: int = 100,
    dt: Optional[float] = None,
    total_time: Optional[float] = None,
    parameter_filter: Optional[Callable[[NumericParam], bool]] = None,
    verbose: bool = True
) -> OptimizationResult:
    """
    Main entry point for HA parameter optimization.

    This function takes an HA specification (as dict or text), extracts all
    numeric parameters, and uses gradient-free optimization to find values
    that minimize trajectory error against ground truth data.

    Args:
        ha_input: HA specification (dict or string containing dict)
        npz_file_path: Path to ground truth NPZ file
        budget: Number of optimization evaluations (default: 100)
        dt: Time step override (uses config value if None)
        total_time: Total time override (uses config value if None)
        parameter_filter: Optional function to filter parameters
        verbose: Whether to print progress

    Returns:
        OptimizationResult with original and optimized specifications

    Example:
        result = optimize_ha_parameters(
            ha_input={"automaton": {...}, "config": {...}},
            npz_file_path='data.npz',
            budget=100
        )
        print(result.summary())
        better_ha = result.get_better_spec()
    """
    # Preprocess input (extract dict, fix common issues)
    ha_dict, messages = extract_and_fix_ha_spec(ha_input, auto_fix=True)

    if ha_dict is None:
        raise ValueError(f"Failed to extract HA specification: {messages}")

    if verbose and messages:
        print("Preprocessing messages:")
        for msg in messages:
            print(f"  - {msg}")

    # Create optimizer and run
    optimizer = HAParamOptimizer(
        ha_dict=ha_dict,
        npz_file_path=npz_file_path,
        dt=dt,
        total_time=total_time,
        parameter_filter=parameter_filter
    )

    return optimizer.optimize(budget=budget, verbose=verbose)


# ============================================================================
# Main Execution Block (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HA Parameter Optimizer - Test Suite")
    print("=" * 80)

    # Test 1: Parameter extraction from ball.json
    print("\n1. Testing parameter extraction from ball.json...")

    ball_json_path = os.path.join(_current_dir, 'Dainarx_code/automata/ATVA/ball.json')

    if os.path.exists(ball_json_path):
        with open(ball_json_path, 'r') as f:
            ball_spec = json.load(f)

        # Remove init_state for testing (not part of core HA)
        test_spec = {k: v for k, v in ball_spec.items() if k != 'init_state'}

        param_ha = ParameterizedHA(test_spec)
        print(f"   Extracted {param_ha.get_parameter_count()} parameters:")
        for p in param_ha.parameters:
            print(f"   - {p.path}: value={p.value}, bounds={p.bounds}")
    else:
        print(f"   Skipped: {ball_json_path} not found")

    # Test 2: Reconstruction test
    print("\n2. Testing parameter reconstruction...")

    test_ha = {
        "automaton": {
            "var": "x1, x2",
            "mode": [
                {"id": 1, "eq": "x1[1] = x2[0], x2[1] = -9.8"}
            ],
            "edge": [
                {
                    "direction": "1 -> 1",
                    "condition": "x1 <= 0",
                    "reset": {
                        "x1": [0],
                        "x2": ["-0.9 * x2[0]"]
                    }
                }
            ]
        },
        "config": {"dt": 0.01, "total_time": 10.0}
    }

    param_ha = ParameterizedHA(test_ha)
    print(f"   Extracted {param_ha.get_parameter_count()} parameters:")
    for p in param_ha.parameters:
        print(f"   - {p.path}: value={p.value}, bounds={p.bounds}")

    # Test reconstruction with modified values
    if param_ha.get_parameter_count() > 0:
        new_values = param_ha.get_original_values() * 1.1  # 10% increase
        reconstructed = param_ha.reconstruct(new_values)
        print("\n   Reconstructed HA (with 10% increased parameters):")
        print(f"   Mode eq: {reconstructed['automaton']['mode'][0]['eq']}")
        if reconstructed['automaton']['edge']:
            reset = reconstructed['automaton']['edge'][0].get('reset', {})
            print(f"   Reset x2: {reset.get('x2', 'N/A')}")

    # Test 3: End-to-end optimization (if ground truth exists)
    print("\n3. Testing end-to-end optimization...")

    gt_path = os.path.join(os.path.dirname(_current_dir), 'data_all/ATVA/ball_g/ground_truth_0.npz')

    if os.path.exists(gt_path):
        print(f"   Ground truth file: {gt_path}")

        # Run a quick optimization with small budget
        result = optimize_ha_parameters(
            ha_input=test_ha,
            npz_file_path=gt_path,
            budget=20,  # Small budget for testing
            verbose=True
        )

        print("\n" + result.summary())
    else:
        print(f"   Skipped: Ground truth file not found at {gt_path}")

    print("\n" + "=" * 80)
    print("Test suite complete!")
    print("=" * 80)
