import numpy as np
import pysindy.optimizers as ps_optimizers
from pysindy.feature_library import (
    PolynomialLibrary,       # Polynomial feature library (powers/interactions up to chosen degree)
    FourierLibrary,          # Fourier feature library (sin/cos terms for periodic dynamics)
    CustomLibrary,           # User-defined feature library from custom candidate functions
    GeneralizedLibrary,      # Generalized composition of multiple libraries (with optional tensoring)
    IdentityLibrary,         # Identity feature library (returns raw state variables)
    PDELibrary,              # PDE feature library with derivative terms on a spatiotemporal grid
    ParameterizedLibrary,    # Library for parameterized dynamics with state and parameter variables
    SINDyPILibrary,          # SINDy-PI feature library for implicit dynamics (deprecated in favor of PDE/WeakPDE)
    TensoredLibrary,         # Tensor-product library combining features from multiple sub-libraries
    WeakPDELibrary,          # Weak-form PDE library using integral formulations for noise robustness
    ConcatLibrary,           # Concatenates features from multiple libraries
)
from pysindy.optimizers import (
    BaseOptimizer,           # Base class for PySINDy optimizers
    SR3,                     # Sparse Relaxed Regularized Regression optimizer
    STLSQ,                   # Sequentially Thresholded Least Squares optimizer (classic SINDy default)
    SSR,                     # Stepwise Sparse Regressor (iterative support pruning)
    FROLS,                   # Forward Regression Orthogonal Least Squares optimizer
    EnsembleOptimizer,       # Ensemble wrapper (e.g., bagging/bragging) over base optimizers
    WrappedOptimizer,        # Wrapper to use external sklearn-like regressors as SINDy optimizers
)

from pysindy.differentiation import BaseDifferentiation
from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SpectralDerivative
from pysindy.differentiation import SINDyDerivative
from pysindy.differentiation import SmoothedFiniteDifference

# Optional optimizers in the official PySINDy API.
# They may be unavailable when optional dependencies (e.g., cvxpy) are missing.
ConstrainedSR3 = getattr(ps_optimizers, "ConstrainedSR3", None)  # Requires cvxpy
StableLinearSR3 = getattr(ps_optimizers, "StableLinearSR3", None)  # Requires cvxpy
TrappingSR3 = getattr(ps_optimizers, "TrappingSR3", None)  # Requires cvxpy
SINDyPI = getattr(ps_optimizers, "SINDyPI", None)  # Requires cvxpy
MIOSR = getattr(ps_optimizers, "MIOSR", None)  # Requires pysindy[miosr] (Gurobi backend)
SBR = getattr(ps_optimizers, "SBR", None)  # Requires numpyro / pysindy[sbr]

# Here is an example of how to use the PySINDy library to fit a model to data.
# ── Feature Libraries ──────────────────────────────────────────────────────────
# Polynomial terms to capture higher-order dynamics
poly_lib = PolynomialLibrary(degree=4, include_interaction=True, include_bias=True)

# Trigonometric terms to capture periodic oscillations
fourier_lib = FourierLibrary(n_frequencies=4)

# Custom library: exponential / logarithmic terms
functions = [
    lambda x: np.exp(x),
    lambda x: np.exp(-x),
    lambda x: np.log(np.abs(x) + 1e-10),
    lambda x: x**2,
    lambda x, y: x * y,
]
custom_exp_lib = CustomLibrary(library_functions=functions)

# Combined library
# x0 → polynomial + log trends
# x1, x2 → trigonometric + exponential (oscillatory / noisy)
feature_library = GeneralizedLibrary([poly_lib, custom_exp_lib, fourier_lib])

# ── Optimizer ─────────────────────────────────────────────────────────────────

# SR3 with threshold tunable on validation performance
optimizer = SR3(reg_weight_lam=0.03)

# ── Quick smoke-test (optional) ───────────────────────────────────────────────
if __name__ == "__main__":
    import pysindy as ps

    # Synthetic 3-variable data for verification
    t = np.linspace(0, 10, 500)
    dt = t[1] - t[0]
    x = np.column_stack([
        np.linspace(0, 5, 500),          # x0: trending signal
        np.sin(2 * np.pi * t),           # x1: oscillatory
        np.cos(2 * np.pi * t) + 0.1 * np.random.randn(500),  # x2: noisy oscillation
    ])

    # IMPORTANT: SINDy.__init__ only accepts: optimizer, feature_library, differentiation_method
    # feature_names goes in fit(), NOT __init__()
    # fit() does NOT have a 'quiet' parameter
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)

    # fit(x, t, x_dot=None, u=None, feature_names=None)
    # - u: control input array, shape (n_samples, n_control_features)
    # - feature_names: list of strings for state variables, e.g. ['x1', 'x2']
    model.fit(x, t=dt, feature_names=['x0', 'x1', 'x2'])
    model.print()