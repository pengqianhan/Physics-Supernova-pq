"""
Simple benchmark to show ProcessPoolExecutor speeds up Nevergrad evaluations.
This mirrors the parallel pattern used in optimize_ha_params.py.
"""

from concurrent import futures
import os
import time

import nevergrad as ng


def slow_objective(x: float) -> float:
    # Simulate heavy computation
    time.sleep(0.05)
    return (x - 0.5) ** 2


def run_minimize(num_workers: int, budget: int) -> float:
    optimizer = ng.optimizers.NGOpt(
        parametrization=ng.p.Scalar(lower=-5.0, upper=5.0),
        budget=budget,
        num_workers=num_workers,
    )

    start = time.perf_counter()
    if num_workers > 1:
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            optimizer.minimize(slow_objective, executor=executor, batch_mode=False)
    else:
        optimizer.minimize(slow_objective)
    end = time.perf_counter()
    return end - start


def main() -> None:
    budget = 40
    cpu_count = os.cpu_count() or 1
    workers = max(2, min(8, cpu_count // 4))

    t_serial = run_minimize(num_workers=1, budget=budget)
    t_parallel = run_minimize(num_workers=workers, budget=budget)

    print(f"budget={budget}")
    print(f"serial  (num_workers=1): {t_serial:.2f}s")
    print(f"parallel(num_workers={workers}): {t_parallel:.2f}s")
    if t_parallel > 0:
        print(f"speedup: {t_serial / t_parallel:.2f}x")


if __name__ == "__main__":
    main()
