# Ground Truth 0 Evaluation Analysis

**Threshold**: mean_diff < 0.001
**Total datasets**: 26
**Passing**: 11
**Failing**: 15
**Success rate**: 42.3%

## Summary Statistics

- Datasets with perfect match (mean_diff = 0): 6
- Datasets with clustering errors: 3

---

## Passing Datasets (mean_diff < 0.001)

| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |
|---------|-----------|----------|-----|------------------|------------|
| ATVA/ball | 0.000000e+00 | 0.000000e+00 | 0.0 | 0 | 2/2 |
| ATVA/cell | 0.000000e+00 | 0.000000e+00 | 0.0 | 0 | 2/2 |
| FaMoS/complex_tank | 0.000000e+00 | 0.000000e+00 | 0.0 | 0 | 18/19 |
| linear/linear_1 | 0.000000e+00 | 0.000000e+00 | 0.0 | 0 | 1/1 |
| linear/one_legged_jumper | 0.000000e+00 | 0.000000e+00 | 0.0 | 0 | 3/3 |
| non_linear/simple_non_poly | 0.000000e+00 | 0.000000e+00 | 0.0 | 0 | 1/7 |
| non_linear/lander | 5.179742e-05 | 2.639847e-04 | 0.001 | 0 | 4/4 |
| linear/loop | 7.236274e-05 | 2.221225e-03 | 0.1 | 1 | 2/2 |
| linear/underdamped_system | 6.385763e-04 | 4.893493e-03 | 0.002 | 0 | 3/3 |
| non_linear/oscillator | 7.568851e-04 | 2.729198e-02 | 0.004 | 0 | 13/50 |
| FaMoS/simple_heating_system | 7.992262e-04 | 2.069350e-03 | 0.004 | 0 | 7/22 |

---

## Failing Datasets (mean_diff >= 0.001)

| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |
|---------|-----------|----------|-----|------------------|------------|
| non_linear/spacecraft | 1.231164e-03 | 6.728255e-02 | 0.012 | 0 | 27/30 |
| ATVA/oci | 2.546690e-03 | 8.347211e-03 | 0.002 | 0 | 15/15 |
| ATVA/tanks | 2.805367e-03 | 2.557987e-02 | 0.016 | 0 | 32/50 |
| linear/dc_motor_position_PID | 4.847899e-03 | 1.879858e-02 | 0.004 | 0 | 30/31 |
| linear/complex_underdamped_system | 6.932841e-03 | 1.355810e-01 | 0.0 | 1 | 19/50 |
| FaMoS/multi_room_heating | 2.171113e-02 | 1.126006e-01 | 0.501 | 0 | 37/49 |
| FaMoS/variable_heating_system | 2.439205e-02 | 1.179787e-01 | 0.228 | 0 | 14/50 |
| non_linear/lotkaVolterra | 2.783526e-02 | 9.752026e-02 | 0.203 | 0 | 15/15 |
| FaMoS/two_state_ha | 1.796139e-01 | 4.163519e-01 | 1.12 | 0 | 16/24 |
| FaMoS/three_state_ha | 1.995152e-01 | 4.503231e-01 | 1.12 | 0 | 22/50 |
| linear/two_tank | 2.582147e-01 | 7.625378e-01 | 3.216 | 0 | 11/11 |
| non_linear/simple_non_linear | 2.678781e-01 | 9.070205e-01 | 0.34400000000000003 | 0 | 29/45 |
| FaMoS/buck_converter | 2.694489e-01 | 1.286574e+00 | 4.997 | 0 | 34/50 |
| non_linear/duffing | 3.443114e-01 | 6.184907e-01 | 2.129 | 0 | 18/50 |
| non_linear/sys_bio | 3.614074e-01 | 1.002014e+00 | 1.577 | 1 | 20/28 |

---

## Passing Datasets Details

### ATVA/ball

- **mean_diff**: 0.000000e+00
- **max_diff**: 0.000000e+00
- **tc**: 0.0
- **clustering_error**: 0
- **iterations**: 2/2
- **HA spec**: `evaluation_results_gemini3flash_260129/ATVA/ball/runs/20260125_174104_11ed/best_ha_specification.json`

### ATVA/cell

- **mean_diff**: 0.000000e+00
- **max_diff**: 0.000000e+00
- **tc**: 0.0
- **clustering_error**: 0
- **iterations**: 2/2
- **HA spec**: `evaluation_results_gemini3flash_260129/ATVA/cell/runs/20260125_174104_6cfe/best_ha_specification.json`

### FaMoS/complex_tank

- **mean_diff**: 0.000000e+00
- **max_diff**: 0.000000e+00
- **tc**: 0.0
- **clustering_error**: 0
- **iterations**: 18/19
- **HA spec**: `evaluation_results_gemini3flash_260129/FaMoS/complex_tank/runs/20260125_180534_4462/best_ha_specification.json`

### linear/linear_1

- **mean_diff**: 0.000000e+00
- **max_diff**: 0.000000e+00
- **tc**: 0.0
- **clustering_error**: 0
- **iterations**: 1/1
- **HA spec**: `evaluation_results_gemini3flash_260129/linear/linear_1/runs/20260127_132035_95e8/best_ha_specification.json`

### linear/one_legged_jumper

- **mean_diff**: 0.000000e+00
- **max_diff**: 0.000000e+00
- **tc**: 0.0
- **clustering_error**: 0
- **iterations**: 3/3
- **HA spec**: `evaluation_results_gemini3flash_260129/linear/one_legged_jumper/runs/20260127_142225_2051/best_ha_specification.json`

### non_linear/simple_non_poly

- **mean_diff**: 0.000000e+00
- **max_diff**: 0.000000e+00
- **tc**: 0.0
- **clustering_error**: 0
- **iterations**: 1/7
- **HA spec**: `evaluation_results_gemini3flash_260129/non_linear/simple_non_poly/runs/20260128_101354_b6a3/best_ha_specification.json`

### non_linear/lander

- **mean_diff**: 5.179742e-05
- **max_diff**: 2.639847e-04
- **tc**: 0.001
- **clustering_error**: 0
- **iterations**: 4/4
- **HA spec**: `evaluation_results_gemini3flash_260129/non_linear/lander/runs/20260127_175716_18f1/best_ha_specification.json`

### linear/loop

- **mean_diff**: 7.236274e-05
- **max_diff**: 2.221225e-03
- **tc**: 0.1
- **clustering_error**: 1
- **iterations**: 2/2
- **HA spec**: `evaluation_results_gemini3flash_260129/linear/loop/runs/20260127_134200_a220/best_ha_specification.json`

### linear/underdamped_system

- **mean_diff**: 6.385763e-04
- **max_diff**: 4.893493e-03
- **tc**: 0.002
- **clustering_error**: 0
- **iterations**: 3/3
- **HA spec**: `evaluation_results_gemini3flash_260129/linear/underdamped_system/runs/20260127_161006_a9a6/best_ha_specification.json`

### non_linear/oscillator

- **mean_diff**: 7.568851e-04
- **max_diff**: 2.729198e-02
- **tc**: 0.004
- **clustering_error**: 0
- **iterations**: 13/50
- **HA spec**: `evaluation_results_gemini3flash_260129/non_linear/oscillator/runs/20260127_193518_3bc4/best_ha_specification.json`

### FaMoS/simple_heating_system

- **mean_diff**: 7.992262e-04
- **max_diff**: 2.069350e-03
- **tc**: 0.004
- **clustering_error**: 0
- **iterations**: 7/22
- **HA spec**: `evaluation_results_gemini3flash_260129/FaMoS/simple_heating_system/runs/20260126_004441_663a/best_ha_specification.json`
