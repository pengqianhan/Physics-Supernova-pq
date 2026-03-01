"""
Hybrid Automaton Parameter Optimization using Nevergrad + LLM-based Parameter Extraction

This script provides a generalized approach to optimize numeric parameters in ANY
Hybrid Automaton JSON specification. It uses:
1. LLM (Gemini via LiteLLM) to automatically extract tunable parameters
2. Nevergrad for gradient-free optimization
3. HAEvaluator for trajectory comparison

Example usage:
    python optimize_ha_params.py  # Runs bouncing ball demo

    # Programmatic usage:
    from optimize_ha_params import optimize_ha_parameters_generic
    result = optimize_ha_parameters_generic(
        ha_spec=your_ha_dict,
        input_data_path="data_all/ATVA/ball"
    )
"""

import hashlib
import json
import math
import os
import re
import sys
import functools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Nevergrad for optimization
import nevergrad as ng
from loky import ProcessPoolExecutor

# Pydantic for structured LLM output
from pydantic import BaseModel, Field
from sqlalchemy import true

# LiteLLM for Gemini API calls
try:
    import litellm
    from dotenv import load_dotenv
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    print("Warning: litellm not installed. LLM-based parameter extraction unavailable.")

# Add path for HA evaluation imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils/Dainarx_code'))
from HA_evaluation import HAEvaluator


# ============================================================================
# Pydantic Schemas for LLM Structured Output
# ============================================================================

class ExtractedParameter(BaseModel):
    """A single extracted parameter from the HA specification."""
    name: str = Field(..., description="Descriptive name, e.g., 'gravity', 'restitution_coefficient'")
    value: float = Field(..., description="Current numeric value in the spec")
    location: str = Field(..., description="Where it appears: 'mode_eq', 'edge_condition', or 'edge_reset'")
    context: str = Field(..., description="The expression containing this parameter, e.g., 'x2[1] = -8.0'")
    lower_bound: float = Field(..., description="Suggested lower bound for optimization")
    upper_bound: float = Field(..., description="Suggested upper bound for optimization")
    description: str = Field(..., description="Physical interpretation of this parameter")


class ParameterExtractionResult(BaseModel):
    """Result of LLM parameter extraction."""
    parameters: List[ExtractedParameter] = Field(..., description="List of all extracted parameters")
    parameterized_spec: str = Field(..., description="HA spec JSON with {param_0}, {param_1}, etc. placeholders")
    budget: int = Field(..., description="Suggested optimization budget (number of evaluations) based on parameter count and search space complexity")


# ============================================================================
# LLM-based Parameter Extraction
# ============================================================================

def _get_parameter_extraction_prompt() -> str:
    """System prompt for LLM parameter extraction."""
    return """You are an expert at analyzing Hybrid Automaton (HA) specifications.

Your task: Systematically replace ALL numeric coefficients in the HA spec with {param_N} placeholders (N is 0-indexed, assigned sequentially across ALL locations).

You MUST parameterize the following THREE locations, and ONLY these three:

═══════════════════════════════════════════════════════════
LOCATION 1: mode.eq — ODE equations (RIGHT-HAND SIDE ONLY)
═══════════════════════════════════════════════════════════
For each term on the RIGHT side of '=', EXCEPT terms that are purely the input variable 'u' alone:
- Replace the numeric coefficient in front of each term with {param_N}.
- If a term has NO explicit coefficient (implicit coefficient of 1), you MUST insert "{param_N} * " in front of it.
- If a term has a negative sign with no explicit coefficient (implicit -1), use "-{param_N} * ".
- The input variable 'u' standing alone (without any coefficient) is kept as-is. But if u has a coefficient like "0.5 * u", replace the 0.5 with {param_N}.
- Exponents (powers) are NOT coefficients — do NOT replace them.

Example:
  Original:  "x[2] = u - 0.3 * x[1] + x[0] - 1 * x[0] ** 3"
  Result:    "x[2] = u - {param_0} * x[1] + {param_1} * x[0] - {param_2} * x[0] ** 3"
  Extracted: param_0=0.3, param_1=1.0, param_2=1.0

  Original:  "x[2] = u - 0.5 * x[1] + x[0] - 1 * x[0] ** 3"
  Result:    "x[2] = u - {param_3} * x[1] + {param_4} * x[0] - {param_5} * x[0] ** 3"
  Extracted: param_3=0.5, param_4=1.0, param_5=1.0

  Original:  "x2[1] = -7.8 + 0.5 * x1[0]"
  Result:    "x2[1] = -{param_0} + {param_1} * x1[0]"
  Extracted: param_0=7.8, param_1=0.5

═══════════════════════════════════════════════════════════
LOCATION 2: edge.condition — Guard conditions
═══════════════════════════════════════════════════════════
Replace every numeric literal (thresholds, constants) in the condition with {param_N}.

Example:
  Original:  "abs(x) <= 0.9"
  Result:    "abs(x) <= {param_6}"
  Extracted: param_6=0.9

  Original:  "abs(x) >= 1.2"
  Result:    "abs(x) >= {param_7}"
  Extracted: param_7=1.2

═══════════════════════════════════════════════════════════
LOCATION 3: edge.reset — Reset maps
═══════════════════════════════════════════════════════════
Replace every numeric value in reset expressions with {param_N}.

Example:
  Original:  "x": ["", "x[1] * 0.3"]
  Result:    "x": ["", "x[1] * {param_8}"]
  Extracted: param_8=0.3

  Original:  "x2": ["-0.9 * x2[0]"]
  Result:    "x2": ["-{param_5} * x2[0]"]
  Extracted: param_5=0.9

═══════════════════════════════════════════════════════════
DO NOT PARAMETERIZE (keep exactly as-is):
═══════════════════════════════════════════════════════════
- The entire "config" section (dt, total_time, order, need_reset, dim, other_items, etc.)
- Mode IDs (e.g., "id": 1)
- Variable indices in brackets (e.g., x[0], x[1], x1[0], x2[1])
- Exponents / powers (e.g., the 3 in "x[0] ** 3")
- The direction string in edges (e.g., "1 -> 2")
- The variable names (var, input fields)

═══════════════════════════════════════════════════════════
PARAMETERIZED SPEC FORMAT — CRITICAL RULES:
═══════════════════════════════════════════════════════════
- {param_N} indices MUST be sequential starting from 0, assigned in order: first all modes (mode 1 eq left-to-right, mode 2 eq left-to-right, ...), then all edges (edge 1 condition, edge 1 reset, edge 2 condition, edge 2 reset, ...)
- Keep sign operators OUTSIDE the placeholder: write "-{param_0}" NOT "{param_0}" for a negative value
- The "parameterized_spec" field MUST be a single valid JSON object string (no extra text, no markdown)
- Use double quotes for all JSON strings

BOUNDS ESTIMATION — USE WIDE RANGES to avoid excluding the true solution:
- For ODE coefficients (damping, stiffness, etc.): use [max(0.01, value * 0.1), max(value * 5.0, 2.0)]
  Example: value=0.3 → lower=0.03, upper=2.0; value=9.8 → lower=0.98, upper=49.0
- For reset/restitution coefficients: ALWAYS use [0.01, 2.0] regardless of current value,
  because restitution can range from near-zero (heavy damping) to >1 (energy gain)
- For guard/threshold values: use [max(0.01, value * 0.2), value * 5.0]
  Example: value=0.9 → lower=0.18, upper=4.5
- General rule: when in doubt, use WIDER bounds. It is much better to have
  a large search space than to accidentally exclude the true parameter value.

═══════════════════════════════════════════════════════════
BUDGET ESTIMATION — Optimization evaluation budget
═══════════════════════════════════════════════════════════
You MUST suggest an optimization budget (number of function evaluations) in the "budget" field.
The budget should scale with the number of parameters and the difficulty of the search space.

Background (from Nevergrad/CMA-ES literature):
- NGOpt meta-optimizer requires at least 12×d evaluations to enable advanced strategies
- CMA-ES considers "moderate budget" as ≥100×d evaluations
- For full CMA convergence, Nevergrad docs recommend 1000×d evaluations
- However, each evaluation involves ODE simulation, so we balance precision vs. compute

Rules:
- Base rule: budget = N_params * 500  (minimum 500)
- If bounds are very wide (upper/lower ratio > 10 for most params), multiply by 1.5
- If there are many parameters (>8), multiply by 1.5 additionally
- Cap at 10000 to avoid excessive computation
- Round to the nearest 100

Examples:
  2 parameters, moderate bounds → budget = 1000
  5 parameters, moderate bounds → budget = 2500
  5 parameters, wide bounds    → budget = 3800
  10 parameters, moderate bounds → budget = 5000
  10 parameters, wide bounds   → budget = 7500
  15 parameters, wide bounds   → budget = 10000

IMPORTANT: The parameterized_spec field must contain ONLY the JSON object, nothing else."""


def extract_parameters_with_llm(
    ha_spec: Dict[str, Any],
    model_id: str = "gemini/gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
    timeout: int = 60,
    verbose: bool = True
) -> Tuple[Optional[ParameterExtractionResult], str]:
    """
    Use LLM to extract tunable parameters from a Hybrid Automaton specification.

    Args:
        ha_spec: The HA specification dictionary
        model_id: LiteLLM model identifier (default: Gemini flash-lite)
        api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
        timeout: API call timeout in seconds
        verbose: Whether to print progress messages

    Returns:
        Tuple of (ParameterExtractionResult or None, status_message)
    """
    if not HAS_LITELLM:
        return None, "Error: litellm not installed"

    # Load API key
    if api_key is None:
        load_dotenv(override=True)
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "Error: No GEMINI_API_KEY found in environment"

    # Prepare the HA spec as formatted JSON
    ha_spec_str = json.dumps(ha_spec, indent=2)

    user_prompt = f"""Analyze this Hybrid Automaton specification and extract all tunable numeric parameters:

```json
{ha_spec_str}
```

Extract parameters, suggest optimization bounds, and create a parameterized version with {{param_N}} placeholders."""

    if verbose:
        print("[LLM] Extracting parameters from HA specification...")

    try:
        response = litellm.completion(
            model=model_id,
            messages=[
                {"role": "system", "content": _get_parameter_extraction_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            api_key=api_key,
            timeout=timeout,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ParameterExtractionResult",
                    "schema": ParameterExtractionResult.model_json_schema(),
                    "strict": True,
                }
            }
        )

        content = response.choices[0].message.content
        if content is None:
            return None, "Error: LLM returned empty response"

        result = ParameterExtractionResult.model_validate_json(content)

        # Widen bounds programmatically to ensure sufficient search range
        for p in result.parameters:
            val = abs(p.value) if p.value != 0 else 1.0
            if 'reset' in p.location or 'restitution' in p.name.lower():
                # Reset/restitution coefficients: always [0.01, 2.0]
                min_lo, min_hi = 0.01, 2.0
            else:
                # General: at least [value*0.1, value*5] with floor of 2.0 on upper
                min_lo = max(0.01, val * 0.1)
                min_hi = max(val * 5.0, 2.0)
            p.lower_bound = min(p.lower_bound, min_lo)
            p.upper_bound = max(p.upper_bound, min_hi)

        # Validate and clean up parameterized spec
        param_spec = result.parameterized_spec.strip()

        # Try to extract just the JSON object if there's extra text
        if not param_spec.startswith('{'):
            # Find the first '{'
            start = param_spec.find('{')
            if start != -1:
                param_spec = param_spec[start:]

        # Find matching closing brace
        brace_count = 0
        end_idx = 0
        for i, c in enumerate(param_spec):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx > 0:
            param_spec = param_spec[:end_idx]

        # Validate it parses as JSON
        try:
            json.loads(param_spec)
            result.parameterized_spec = param_spec
        except json.JSONDecodeError as e:
            if verbose:
                print(f"[LLM] Warning: Parameterized spec has JSON errors: {e}")
                print(f"[LLM] Raw spec: {param_spec[:500]}...")

        if verbose:
            print(f"[LLM] ✓ Extracted {len(result.parameters)} parameters (suggested budget: {result.budget}):")
            for i, p in enumerate(result.parameters):
                print(f"      {i}: {p.name} = {p.value} (bounds: [{p.lower_bound}, {p.upper_bound}])")
            print(f"\n[LLM] Parameterized spec preview:")
            try:
                preview_spec = json.loads(result.parameterized_spec)
                for mode in preview_spec.get('automaton', {}).get('mode', []):
                    print(f"      Mode {mode.get('id')}: {mode.get('eq')}")
                for edge in preview_spec.get('automaton', {}).get('edge', []):
                    print(f"      Edge {edge.get('direction')}:")
                    print(f"      Edge reset: {edge.get('reset')}")
                    print(f"      Edge condition: {edge.get('condition')}")
            except Exception as parse_err:
                print(f"      Parse error: {parse_err}")
                print(f"      Raw: {result.parameterized_spec[:300]}...")

        return result, "Success"

    except Exception as e:
        error_msg = f"LLM extraction failed: {type(e).__name__}: {str(e)[:200]}"
        if verbose:
            print(f"[LLM] ✗ {error_msg}")
        return None, error_msg


# ============================================================================
# Extraction Result Caching
# ============================================================================

def _get_cache_path(ha_spec: Dict[str, Any], cache_dir: str = ".extraction_cache") -> str:
    """Get cache file path based on a hash of the HA spec."""
    spec_str = json.dumps(ha_spec, sort_keys=True)
    spec_hash = hashlib.md5(spec_str.encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"extraction_{spec_hash}.json")


def _save_extraction_result(result: ParameterExtractionResult, cache_path: str, verbose: bool = True) -> None:
    """Save extraction result to a local JSON cache file."""
    with open(cache_path, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    if verbose:
        print(f"[Cache] Saved extraction result to {cache_path}")


def _load_extraction_result(cache_path: str, verbose: bool = True) -> Optional[ParameterExtractionResult]:
    """Load extraction result from a local JSON cache file if it exists."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        result = ParameterExtractionResult.model_validate(data)
        if verbose:
            print(f"[Cache] Loaded extraction result from {cache_path} ({len(result.parameters)} parameters)")
        return result
    except Exception as e:
        if verbose:
            print(f"[Cache] Failed to load cache ({e}), will re-extract with LLM")
        return None


# ============================================================================
# Parameterization and Reconstruction
# ============================================================================

def reconstruct_ha_spec(
    parameterized_spec_str: str,
    params: np.ndarray,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Reconstruct a concrete HA specification from a parameterized template.

    Args:
        parameterized_spec_str: JSON string with {param_N} placeholders
        params: Array of parameter values
        verbose: Whether to print debug info

    Returns:
        Reconstructed HA specification dictionary
    """
    # Build substitution dictionary
    substitutions = {f"param_{i}": str(params[i]) for i in range(len(params))}

    # Perform substitution
    concrete_spec_str = parameterized_spec_str
    for key, value in substitutions.items():
        concrete_spec_str = concrete_spec_str.replace("{" + key + "}", value)

    if verbose:
        print(f"[Reconstruct] Substituted {len(params)} parameters")

    # Parse and return
    return json.loads(concrete_spec_str)


# ============================================================================
# Objective Function
# ============================================================================

def create_objective_function(
    parameterized_spec_str: str,
    npz_file_paths: List[str],
    dt: float,
    total_time: float,
    verbose: bool = False
):
    """
    Create an objective function for Nevergrad optimization.

    Args:
        parameterized_spec_str: JSON string with {param_N} placeholders
        npz_file_paths: List of paths to ground truth NPZ files (uses average mean_diff)
        dt: Time step for simulation
        total_time: Total simulation time
        verbose: Whether to print intermediate results

    Returns:
        Callable objective function that takes params array and returns error
    """
    def objective(params: np.ndarray) -> float:
        try:
            # Reconstruct HA spec with current parameters
            ha_spec = reconstruct_ha_spec(parameterized_spec_str, params, verbose=False)

            # Compute mean_diff for each NPZ file and average
            mean_diffs = []
            for npz_path in npz_file_paths:
                evaluator = HAEvaluator(
                    ha_dict=ha_spec,
                    npz_file_path=npz_path,
                    dt=dt,
                    total_time=total_time
                )

                evaluator.load_ground_truth()
                evaluator.simulate()
                metrics = evaluator.compute_metrics()

                mean_diff = metrics.get('mean_diff', float('inf'))
                if mean_diff is None:
                    mean_diff = float('inf')
                mean_diffs.append(mean_diff)

            # Compute average mean_diff across all files
            avg_mean_diff = float(np.mean(mean_diffs)) if mean_diffs else float('inf')

            if verbose:
                param_str = ", ".join([f"{p:.4f}" for p in params])
                if len(npz_file_paths) > 1:
                    diffs_str = ", ".join([f"{d:.6f}" for d in mean_diffs])
                    print(f"  params=[{param_str}] -> mean_diffs=[{diffs_str}] avg={avg_mean_diff:.6f}")
                else:
                    print(f"  params=[{param_str}] -> mean_diff={avg_mean_diff:.6f}")

            return avg_mean_diff

        except Exception as e:
            if verbose:
                print(f"  params={params} -> ERROR: {e}")
            return float('inf')

    return objective


def _nevergrad_objective(
    *,
    parameterized_spec_str: str,
    npz_file_paths: List[str],
    dt: float,
    total_time: float,
    verbose: bool,
    param_count: int,
    **kwargs
) -> float:
    """Top-level objective for multiprocessing compatibility."""
    params = np.array([kwargs[f"param_{i}"] for i in range(param_count)])
    try:
        # Reconstruct HA spec with current parameters
        ha_spec = reconstruct_ha_spec(parameterized_spec_str, params, verbose=False)

        # Compute mean_diff for each NPZ file and average
        mean_diffs = []
        for npz_path in npz_file_paths:
            evaluator = HAEvaluator(
                ha_dict=ha_spec,
                npz_file_path=npz_path,
                dt=dt,
                total_time=total_time
            )

            evaluator.load_ground_truth()
            evaluator.simulate()
            metrics = evaluator.compute_metrics()

            mean_diff = metrics.get('mean_diff', float('inf'))
            if mean_diff is None:
                mean_diff = float('inf')
            mean_diffs.append(mean_diff)

        # Compute average mean_diff across all files
        avg_mean_diff = float(np.mean(mean_diffs)) if mean_diffs else float('inf')

        if verbose:
            param_str = ", ".join([f"{p:.4f}" for p in params])
            if len(npz_file_paths) > 1:
                diffs_str = ", ".join([f"{d:.6f}" for d in mean_diffs])
                print(f"  params=[{param_str}] -> mean_diffs=[{diffs_str}] avg={avg_mean_diff:.6f}")
            else:
                print(f"  params=[{param_str}] -> mean_diff={avg_mean_diff:.6f}")

        return avg_mean_diff

    except Exception as e:
        if verbose:
            print(f"  params={params} -> ERROR: {e}")
        return float('inf')


# ============================================================================
# Generic Optimization Function
# ============================================================================

def optimize_ha_parameters_generic(
    ha_spec: Dict[str, Any],
    input_data_path: str,
    model_id: str = "gemini/gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
    verbose: bool = True,
    train_num: int = 1,
    num_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimize parameters in ANY Hybrid Automaton specification using LLM extraction + Nevergrad.

    This is the main entry point for generic HA parameter optimization.
    The optimization budget is automatically determined by the LLM based on
    the number and complexity of extracted parameters.

    Args:
        ha_spec: The HA specification dictionary (any valid HA JSON)
        input_data_path: Path to data directory (e.g., "data_all/ATVA/ball").
                         Ground truth files are found in "{input_data_path}_g/" directory.
        model_id: LLM model for parameter extraction
        api_key: Gemini API key (reads from env if None)
        verbose: Whether to print progress
        train_num: Number of ground truth files to use (averages mean_diff across [:train_num] files)
        num_workers: Parallel worker count for Nevergrad evaluation.
            If None, uses a conservative shared-server default.

    Returns:
        Dictionary containing:
        - 'parameters': Dict mapping parameter names to optimal values
        - 'error': Final optimization error (mean_diff)
        - 'ha_spec': Optimized HA specification
        - 'extraction_result': Raw LLM extraction result
    """
    if verbose:
        print("=" * 60)
        print("Generic HA Parameter Optimization")
        print("=" * 60)

    # Step 1: Extract parameters using LLM (with local caching)
    if verbose:
        print("\n[Step 1] Extracting parameters with LLM...")

    cache_path = _get_cache_path(ha_spec)
    extraction_result = _load_extraction_result(cache_path, verbose=verbose)

    if extraction_result is None:
        extraction_result, status = extract_parameters_with_llm(
            ha_spec=ha_spec,
            model_id=model_id,
            api_key=api_key,
            verbose=verbose
        )

        if extraction_result is None:
            raise RuntimeError(f"Parameter extraction failed: {status}")

        _save_extraction_result(extraction_result, cache_path, verbose=verbose)

    if len(extraction_result.parameters) == 0:
        raise RuntimeError("No parameters found to optimize")

    # Use LLM-suggested budget (designed for num_workers=1)
    base_budget = extraction_result.budget
    if verbose:
        print(f"[LLM] Suggested base budget (for sequential): {base_budget}")

    # Step 2: Build Nevergrad parametrization
    if verbose:
        print(f"\n[Step 2] Building Nevergrad optimizer with {len(extraction_result.parameters)} parameters...")

    # Create parameter specification for Nevergrad
    ng_params = {}
    param_names = []
    for i, p in enumerate(extraction_result.parameters):
        param_name = f"param_{i}"
        param_names.append(p.name)
        ng_params[param_name] = ng.p.Scalar(lower=p.lower_bound, upper=p.upper_bound)
        if verbose:
            print(f"    {param_name} ({p.name}): [{p.lower_bound}, {p.upper_bound}]")

    parametrization = ng.p.Instrumentation(**ng_params)

    # Create optimizer
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        # Default: use 1/4 of CPUs, capped at 16 for good speed/accuracy balance
        num_workers = max(1, min(16, cpu_count // 4))

    # Scale budget by sqrt(num_workers) to compensate for parallel information loss.
    # Theory (Nevergrad/CMA-ES): parallel batches waste ~sqrt(N) evaluations,
    # so we need sqrt(N) more total evaluations to match sequential accuracy.
    # Net wall-clock speedup: num_workers / sqrt(num_workers) = sqrt(num_workers).
    if num_workers > 1:
        budget = int(base_budget * math.sqrt(num_workers))
        if verbose:
            print(f"[Parallel] num_workers={num_workers}, "
                  f"adjusted budget: {base_budget} × √{num_workers} = {budget}")
            print(f"[Parallel] Expected wall-clock speedup: ~{math.sqrt(num_workers):.1f}x")
    else:
        budget = base_budget

    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization,
        budget=budget,
        num_workers=num_workers
    )

    # Step 3: Get config values for simulation
    dt = ha_spec.get('config', {}).get('dt', 0.001)
    total_time = ha_spec.get('config', {}).get('total_time', 10.0)

    # Step 4: Resolve NPZ file paths from input_data_path + '_g' directory
    # Ground truth files are in a folder with "_g" suffix (e.g., "data_all/ATVA/ball_g")
    ground_truth_path = input_data_path.rstrip('/') + '_g'

    if not os.path.isdir(ground_truth_path):
        raise RuntimeError(f"Ground truth directory not found: {ground_truth_path}")

    # Find ground_truth_*.npz files
    gt_files = [f for f in os.listdir(ground_truth_path)
                if f.startswith('ground_truth') and f.endswith('.npz')]

    if not gt_files:
        raise RuntimeError(f"No ground_truth_*.npz files found in: {ground_truth_path}")

    # Sort by numeric index (ground_truth_0.npz, ground_truth_1.npz, ...)
    def extract_index(filename: str) -> float:
        match = re.search(r'ground_truth_(\d+)\.npz', filename)
        return int(match.group(1)) if match else float('inf')

    gt_files.sort(key=extract_index)

    # Use [:train_num] files
    npz_file_paths = [os.path.join(ground_truth_path, f) for f in gt_files[:train_num]]

    if not npz_file_paths:
        raise RuntimeError(f"No ground truth NPZ files found at: {ground_truth_path}")

    if verbose:
        print(f"\n[Step 3] Using {len(npz_file_paths)} ground truth file(s) for optimization:")
        for p in npz_file_paths:
            print(f"    - {p}")

    # Step 4: Run optimization
    if verbose:
        print(f"\n[Step 4] Running optimization (budget={budget}, num_workers={num_workers})...")
        print("-" * 60)

    objective_fn = functools.partial(
        _nevergrad_objective,
        parameterized_spec_str=extraction_result.parameterized_spec,
        npz_file_paths=npz_file_paths,
        dt=dt,
        total_time=total_time,
        verbose=verbose,
        param_count=len(extraction_result.parameters)
    )

    if optimizer.num_workers > 1:
        with ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(
                objective_fn,
                executor=executor,
                batch_mode=False
            )
    else:
        recommendation = optimizer.minimize(objective_fn)

    # Step 6: Extract results
    optimal_params = {
        extraction_result.parameters[i].name: recommendation.kwargs[f"param_{i}"]
        for i in range(len(extraction_result.parameters))
    }
    optimal_error = recommendation.loss

    # Reconstruct optimal HA spec
    optimal_params_array = np.array([recommendation.kwargs[f"param_{i}"]
                                      for i in range(len(extraction_result.parameters))])
    optimal_ha_spec = reconstruct_ha_spec(
        extraction_result.parameterized_spec,
        optimal_params_array
    )

    if verbose:
        print("-" * 60)
        print("\n[Results]")
        print("Optimal parameters:")
        for name, value in optimal_params.items():
            print(f"    {name}: {value:.6f}")
        print(f"Final error (mean_diff): {optimal_error:.6f}")

    return {
        'parameters': optimal_params,
        'error': optimal_error,
        'ha_spec': optimal_ha_spec,
        'extraction_result': extraction_result
    }


# ============================================================================
# Example Usage / Demo
# ============================================================================

if __name__ == "__main__":
    # Path to data directory (ground truth will be found in data_all/ATVA/ball_g/)
    input_data_path = "data_all/non_linear/duffing"
    ground_truth_dir = input_data_path + '_g'

    # Check if directory exists
    if not os.path.isdir(ground_truth_dir):
        print(f"Error: Ground truth directory not found: {ground_truth_dir}")
        print("Please ensure the data_all/ATVA/ball_g/ directory exists with ground truth files.")
        sys.exit(1)

    print("=" * 70)
    print("Generic Hybrid Automaton Parameter Optimization Demo")
    print("=" * 70)
    print()
    print("This demo shows how to optimize parameters in ANY HA specification")
    print("using LLM-based parameter extraction + Nevergrad optimization.")
    print()

    # Create a bouncing ball HA spec with WRONG initial parameters
    # The LLM will extract these and Nevergrad will optimize them
    # Ground truth: gravity=9.8, restitution=0.9
    initial_ha_spec = {
  "automaton": {
    "var": "x",
    "input": "u",
    "mode": [
      {
        "id": 1,
        "eq": "x[2] = u - 0.3 * x[1] + x[0] - 1* x[0] ** 3"
      },
      {
        "id": 2,
        "eq": "x[2] = u - 0.5 * x[1] + x[0] - 1* x[0] ** 3"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 2",
        "condition": "abs(x) <= 0.9",
        "reset": {
          "x": ["", "x[1] * 0.3"]
        }
      },
      {
        "direction": "2 -> 1",
        "condition": "abs(x) >= 1.2",
        "reset": {
          "x": ["", "x[1] * 0.95"]
        }
      }
    ]
  },
  "config": {
    "dt": 0.001,
    "total_time": 10.0,
    "order": 2,
    "need_reset": True
  }
}

    print("Initial HA Specification (with WRONG parameters):")
    print("-" * 70)
    print(json.dumps(initial_ha_spec, indent=2))
    print()

    print("Ground truth parameters (for reference):")
    print("  gravity = 9.8 m/s²")
    print("  restitution = 0.9")
    print()

    # Run generic optimization
    try:
        result = optimize_ha_parameters_generic(
            ha_spec=initial_ha_spec,
            input_data_path=input_data_path,
            model_id="gemini/gemini-3-flash-preview",  # Per CLAUDE.md: use flash-lite for testing
            verbose=True,
            train_num=1  # Use first ground truth file
        )

        print()
        print("=" * 70)
        print("FINAL OPTIMIZED HA SPECIFICATION:")
        print("=" * 70)
        print(json.dumps(result['ha_spec'], indent=2))

        print()
        print("Comparison with ground truth:")
        print("-" * 40)
        for name, value in result['parameters'].items():
            if 'gravity' in name.lower():
                print(f"  {name}: {value:.4f} (true: 9.8)")
            elif 'restitution' in name.lower():
                print(f"  {name}: {value:.4f} (true: 0.9)")
            else:
                print(f"  {name}: {value:.4f}")

    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
