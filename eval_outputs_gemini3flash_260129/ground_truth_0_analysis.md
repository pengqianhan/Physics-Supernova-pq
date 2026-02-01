# Ground Truth 0 Evaluation Analysis

**Threshold**: mean_diff < 0.001
**Total datasets**: 26
**Passing**: 11
**Failing**: 15

## Passing Datasets (mean_diff < 0.001)

| Dataset | mean_diff | max_diff | status |
|---------|-----------|----------|--------|
| ATVA/ball | 0.000000 | 0.000000 | success |
| ATVA/cell | 0.000000 | 0.000000 | success |
| FaMoS/complex_tank | 0.000000 | 0.000000 | success |
| FaMoS/simple_heating_system | 0.000799 | 0.002069 | success |
| linear/linear_1 | 0.000000 | 0.000000 | success |
| linear/loop | 0.000072 | 0.002221 | success |
| linear/one_legged_jumper | 0.000000 | 0.000000 | success |
| linear/underdamped_system | 0.000639 | 0.004893 | success |
| non_linear/lander | 0.000052 | 0.000264 | success |
| non_linear/oscillator | 0.000757 | 0.027292 | success |
| non_linear/simple_non_poly | 0.000000 | 0.000000 | success |

## Failing Datasets (mean_diff >= 0.001)

| Dataset | mean_diff | max_diff | status |
|---------|-----------|----------|--------|
| ATVA/oci | 0.002547 | 0.008347 | success |
| ATVA/tanks | 0.002805 | 0.025580 | success |
| FaMoS/buck_converter | 0.269449 | 1.286574 | success |
| FaMoS/multi_room_heating | 0.021711 | 0.112601 | success |
| FaMoS/three_state_ha | 0.199515 | 0.450323 | success |
| FaMoS/two_state_ha | 0.179614 | 0.416352 | success |
| FaMoS/variable_heating_system | 0.024392 | 0.117979 | success |
| linear/complex_underdamped_system | 0.006933 | 0.135581 | success |
| linear/dc_motor_position_PID | 0.004848 | 0.018799 | success |
| linear/two_tank | 0.258215 | 0.762538 | success |
| non_linear/duffing | 0.344311 | 0.618491 | success |
| non_linear/lotkaVolterra | 0.027835 | 0.097520 | success |
| non_linear/simple_non_linear | 0.267878 | 0.907020 | success |
| non_linear/spacecraft | 0.001231 | 0.067283 | success |
| non_linear/sys_bio | 0.361407 | 1.002014 | success |