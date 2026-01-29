# Dataset Classification by Mean Difference (Gemini 3 Flash)

## Summary

- **Category 1 (Excellent)**: 4 datasets ($\bar{\epsilon} \leq 0.001$)
- **Category 2 (Good)**: 11 datasets ($0.001 < \bar{\epsilon} \leq 0.1$)
- **Category 3 (Poor)**: 11 datasets ($\bar{\epsilon} > 0.1$)

## Category 1: Excellent ($\bar{\epsilon} \leq 0.001$)

| Dataset | Mean Diff | Max Diff | TC Mean | Max Iter |
|---------|-----------|----------|---------|------------|
| ATVA/ball | 0 | 0 | 0.000 | 2 |
| linear/one_legged_jumper | 0 | 0 | 0.000 | 3 |
| ATVA/cell | 1.26e-05 | 1.26e-05 | 0.000 | 2 |
| ATVA/oci | 0.0010 | 0.0031 | 0.001 | 15 |

## Category 2: Good ($0.001 < \bar{\epsilon} \leq 0.1$)

| Dataset | Mean Diff | Max Diff | TC Mean | Max Iter |
|---------|-----------|----------|---------|------------|
| non_linear/lander | 0.0010 | 0.0042 | 0.000 | 4 |
| non_linear/spacecraft | 0.0014 | 0.055 | 0.010 | 30 |
| linear/loop | 0.0018 | 0.070 | 0.792 | 2 |
| non_linear/oscillator | 0.0031 | 0.054 | 0.005 | 50 |
| FaMoS/complex_tank | 0.0099 | 0.199 | 0.463 | 19 |
| linear/complex_underdamped_system | 0.016 | 0.210 | 1.166 | 50 |
| FaMoS/simple_heating_system | 0.016 | 0.028 | 0.018 | 22 |
| non_linear/lotkaVolterra | 0.019 | 0.063 | 0.440 | 15 |
| FaMoS/multi_room_heating | 0.047 | 0.188 | 0.526 | 49 |
| linear/two_tank | 0.066 | 0.208 | 0.190 | 11 |
| non_linear/simple_non_linear | 0.069 | 0.268 | 0.194 | 45 |

## Category 3: Poor ($\bar{\epsilon} > 0.1$)

| Dataset | Mean Diff | Max Diff | TC Mean | Max Iter |
|---------|-----------|----------|---------|------------|
| FaMoS/variable_heating_system | 0.109 | 0.322 | 0.522 | 50 |
| non_linear/duffing | 0.138 | 0.362 | 0.757 | 50 |
| linear/dc_motor_position_PID | 0.145 | 0.297 | 0.727 | 31 |
| FaMoS/two_state_ha | 0.194 | 0.393 | 1.045 | 24 |
| linear/linear_1 | 0.217 | 0.488 | 0.463 | 1 |
| ATVA/tanks | 0.223 | 0.876 | 0.402 | 50 |
| linear/underdamped_system | 0.225 | 0.486 | 0.670 | 3 |
| FaMoS/three_state_ha | 0.246 | 0.595 | 1.817 | 50 |
| non_linear/simple_non_poly | 0.251 | 0.379 | 1.849 | 7 |
| FaMoS/buck_converter | 0.367 | 1.624 | 4.997 | 50 |
| non_linear/sys_bio | 0.368 | 1.516 | 1.524 | 28 |