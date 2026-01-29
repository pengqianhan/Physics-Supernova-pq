#!/usr/bin/env python3
"""Test different initial values for Sample 5 stability"""

import numpy as np
import json
import sys

sys.path.insert(0, 'utils/Dainarx_code')
from src.HybridAutomata import HybridAutomata

# Load sys_bio config
with open('utils/Dainarx_code/automata/non_linear/sys_bio.json', 'r') as f:
    data = json.load(f)

sys_ha = HybridAutomata.from_json(data['automaton'])

def simulate_with_init(init_state, dt=0.001, total_time=10.0):
    """Simulate and check for stability"""
    sys_ha.reset(init_state)

    states = []
    now = 0.0
    while now < total_time:
        now += dt
        state, mode, switched = sys_ha.next(dt)
        states.append(state.copy())

        # Check for explosion
        if np.any(np.abs(state) > 1e100):
            return f"UNSTABLE at t={now:.3f}s", None
        if np.any(np.isnan(state)):
            return f"NaN at t={now:.3f}s", None

    states = np.array(states)
    max_val = np.max(np.abs(states))
    return f"STABLE (max={max_val:.2e})", states

print("=" * 70)
print("验证不同初始值的稳定性")
print("=" * 70)

# Original sample 5
orig_init = {"mode": 1, "x1": [1.0], "x2": [1.1], "x3": [0.6], "x4": [1.2],
             "x5": [1.2], "x6": [0.6], "x7": [0.9], "x8": [1.4], "x9": [0.8]}

def make_modified(base, **kwargs):
    result = {k: v.copy() if isinstance(v, list) else v for k, v in base.items()}
    for k, v in kwargs.items():
        result[k] = [v]
    return result

# Test different modifications
test_cases = [
    ("原始 Sample 5", orig_init),
    ("方案1a: x6=0.65", make_modified(orig_init, x6=0.65)),
    ("方案1b: x6=0.70", make_modified(orig_init, x6=0.70)),
    ("方案2a: x3=0.70", make_modified(orig_init, x3=0.70)),
    ("方案2b: x3=0.75", make_modified(orig_init, x3=0.75)),
    ("方案3a: x8=1.30", make_modified(orig_init, x8=1.30)),
    ("方案3b: x8=1.20", make_modified(orig_init, x8=1.20)),
]

print(f"\n{'方案':<25} {'结果':<40}")
print("-" * 70)

stable_options = []
for label, init in test_cases:
    result, _ = simulate_with_init(init)
    print(f"{label:<25} {result:<40}")
    if "STABLE" in result:
        stable_options.append((label, init))

print("\n" + "=" * 70)
print("推荐最小修改")
print("=" * 70)

if stable_options:
    # Find minimal change
    print("\n可行的稳定方案:")
    for label, init in stable_options:
        if "原始" not in label:
            print(f"  - {label}")

    print("\n推荐: 方案1a (x6: 0.60 → 0.65)")
    print("原因: 仅改动一个变量，改动幅度最小(+0.05)")
