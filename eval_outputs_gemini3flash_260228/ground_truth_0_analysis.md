# Ground Truth 0 Evaluation Analysis

## Classification Criteria

- **Category 1 (Excellent)**: mean_diff < 0.001
- **Category 2 (Good)**: 0.001 <= mean_diff < 0.005
- **Category 3 (Poor)**: mean_diff >= 0.005

## Overview

**Total datasets**: 11
**Excellent (Cat 1)**: 2 (18.2%)
**Good (Cat 2)**: 5 (45.5%)
**Poor (Cat 3)**: 4 (36.4%)

## Summary Statistics

- Datasets with perfect match (mean_diff = 0): 1
- Datasets with clustering errors: 1

---

## Category 1: Excellent (mean_diff < 0.001)

| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |
|---------|-----------|----------|-----|------------------|------------|
| linear/two_tank | 0.000000 | 0.000000 | 0.0 | 0 | 2/2 |
| non_linear/lotkaVolterra | 0.000759 | 0.002597 | 0.01 | 0 | 37/50 |

---

## Category 2: Good (0.001 <= mean_diff < 0.005)

| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |
|---------|-----------|----------|-----|------------------|------------|
| FaMoS/variable_heating_system | 0.003733 | 0.017237 | 0.03 | 0 | 38/50 |
| non_linear/sys_bio | 0.003853 | 0.122730 | 0.004 | 0 | 24/50 |
| FaMoS/multi_room_heating | 0.004098 | 0.016600 | 0.085 | 0 | 32/50 |
| FaMoS/buck_converter | 0.004402 | 0.043031 | 8e-05 | 0 | 26/50 |
| linear/complex_underdamped_system | 0.004859 | 0.092220 | 0.03 | 0 | 35/50 |

---

## Category 3: Poor (mean_diff >= 0.005)

| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |
|---------|-----------|----------|-----|------------------|------------|
| FaMoS/three_state_ha | 0.069637 | 0.679449 | 1.11 | 0 | 16/44 |
| non_linear/duffing | 0.107510 | 0.328050 | 4.642 | 1 | 24/50 |
| non_linear/simple_non_linear | 0.126060 | 0.912440 | 4.261 | 0 | 19/47 |
| FaMoS/two_state_ha | 0.166897 | 0.411551 | 1.11 | 0 | 23/50 |

---

## Excellent Datasets Details

### linear/two_tank

- **mean_diff**: 0.000000
- **max_diff**: 0.000000
- **tc**: 0.0
- **clustering_error**: 0
- **iterations**: 2/2
- **HA spec**: `evaluation_results/linear/two_tank/runs/20260226_005013_6b36/best_ha_specification.json`

### non_linear/lotkaVolterra

- **mean_diff**: 0.000759
- **max_diff**: 0.002597
- **tc**: 0.01
- **clustering_error**: 0
- **iterations**: 37/50
- **HA spec**: `evaluation_results/non_linear/lotkaVolterra/runs/20260224_150643_afe6/best_ha_specification.json`
