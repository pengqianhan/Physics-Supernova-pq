# JSON Automata Configuration Analysis Report

**扫描目录**: `/home/phan635/HybridAutomata/baseline_ha/Physics-Supernova-pq/utils/Dainarx_code/automata`

---

## 1. 基本统计

- **总JSON文件数**: 27
- **包含input字段的文件数**: 4
- **不包含input字段的文件数**: 23
- **包含config字段的文件数**: 27

## 2. Input字段分析

### 包含input字段的文件 (4):

- ✓ `ATVA/tanks.json`
- ✓ `linear/complex_underdamped_system.json`
- ✓ `non_linear/duffing.json`
- ✓ `non_linear/duffing_simulation.json`

### 不包含input字段的文件 (23):

- ✗ `ATVA/ball.json`
- ✗ `ATVA/cell.json`
- ✗ `ATVA/oci.json`
- ✗ `FaMoS/buck_converter.json`
- ✗ `FaMoS/complex_tank.json`
- ✗ `FaMoS/multi_room_heating.json`
- ✗ `FaMoS/simple_heating_system.json`
- ✗ `FaMoS/three_state_ha.json`
- ✗ `FaMoS/two_state_ha.json`
- ✗ `FaMoS/variable_heating_system.json`
- ✗ `linear/dc_motor_position_PID.json`
- ✗ `linear/linear_1.json`
- ✗ `linear/loop.json`
- ✗ `linear/one_legged_jumper.json`
- ✗ `linear/two_tank.json`
- ✗ `linear/underdamped_system.json`
- ✗ `non_linear/lander.json`
- ✗ `non_linear/lotkaVolterra.json`
- ✗ `non_linear/oscillator.json`
- ✗ `non_linear/simple_non_linear.json`
- ✗ `non_linear/simple_non_poly.json`
- ✗ `non_linear/spacecraft.json`
- ✗ `non_linear/sys_bio.json`

## 3. Config参数统计

### 3.1 Order参数

| Order值 | 文件数量 | 文件列表 |
|---------|---------|----------|
| 1 | 19 | `ATVA/ball.json`<br>`ATVA/cell.json`<br>`ATVA/oci.json`<br>`ATVA/tanks.json`<br>`FaMoS/buck_converter.json`<br>`FaMoS/complex_tank.json`<br>`FaMoS/multi_room_heating.json`<br>`FaMoS/simple_heating_system.json`<br>`FaMoS/variable_heating_system.json`<br>`linear/linear_1.json`<br>`linear/one_legged_jumper.json`<br>`linear/two_tank.json`<br>`non_linear/lander.json`<br>`non_linear/lotkaVolterra.json`<br>`non_linear/oscillator.json`<br>`non_linear/simple_non_linear.json`<br>`non_linear/simple_non_poly.json`<br>`non_linear/spacecraft.json`<br>`non_linear/sys_bio.json` |
| 2 | 7 | `FaMoS/three_state_ha.json`<br>`FaMoS/two_state_ha.json`<br>`linear/complex_underdamped_system.json`<br>`linear/loop.json`<br>`linear/underdamped_system.json`<br>`non_linear/duffing.json`<br>`non_linear/duffing_simulation.json` |
| 4 | 1 | `linear/dc_motor_position_PID.json` |

### 3.2 Need_reset参数

| Need_reset值 | 文件数量 | 文件列表 |
|--------------|---------|----------|
| True | 10 | `ATVA/ball.json`<br>`FaMoS/three_state_ha.json`<br>`FaMoS/two_state_ha.json`<br>`linear/complex_underdamped_system.json`<br>`linear/dc_motor_position_PID.json`<br>`linear/loop.json`<br>`linear/underdamped_system.json`<br>`non_linear/duffing.json`<br>`non_linear/duffing_simulation.json`<br>`non_linear/simple_non_poly.json` |
| null | 17 | `ATVA/cell.json`<br>`ATVA/oci.json`<br>`ATVA/tanks.json`<br>`FaMoS/buck_converter.json`<br>`FaMoS/complex_tank.json`<br>`FaMoS/multi_room_heating.json`<br>`FaMoS/simple_heating_system.json`<br>`FaMoS/variable_heating_system.json`<br>`linear/linear_1.json`<br>`linear/one_legged_jumper.json`<br>`linear/two_tank.json`<br>`non_linear/lander.json`<br>`non_linear/lotkaVolterra.json`<br>`non_linear/oscillator.json`<br>`non_linear/simple_non_linear.json`<br>`non_linear/spacecraft.json`<br>`non_linear/sys_bio.json` |

### 3.3 Kernel参数

| Kernel值 | 文件数量 | 文件列表 |
|----------|---------|----------|
| rbf | 6 | `FaMoS/buck_converter.json`<br>`FaMoS/multi_room_heating.json`<br>`non_linear/duffing.json`<br>`non_linear/duffing_simulation.json`<br>`non_linear/lotkaVolterra.json`<br>`non_linear/oscillator.json` |
| null | 21 | `ATVA/ball.json`<br>`ATVA/cell.json`<br>`ATVA/oci.json`<br>`ATVA/tanks.json`<br>`FaMoS/complex_tank.json`<br>`FaMoS/simple_heating_system.json`<br>`FaMoS/three_state_ha.json`<br>`FaMoS/two_state_ha.json`<br>`FaMoS/variable_heating_system.json`<br>`linear/complex_underdamped_system.json`<br>`linear/dc_motor_position_PID.json`<br>`linear/linear_1.json`<br>`linear/loop.json`<br>`linear/one_legged_jumper.json`<br>`linear/two_tank.json`<br>`linear/underdamped_system.json`<br>`non_linear/lander.json`<br>`non_linear/simple_non_linear.json`<br>`non_linear/simple_non_poly.json`<br>`non_linear/spacecraft.json`<br>`non_linear/sys_bio.json` |

### 3.4 所有配置项汇总

在所有JSON文件的config字段中，出现过的所有配置项：

- `class_weight`
- `dt`
- `kernel`
- `minus`
- `need_bias`
- `need_reset`
- `order`
- `other_items`
- `self_loop`
- `svm_c`
- `total_time`
- `window_size`

## 4. 详细配置表

| 文件 | Order | Need_reset | Kernel | 其他配置项 |
|------|-------|------------|--------|------------|
| `ATVA/ball.json` | 1 | True | null | `dt`, `total_time`, `other_items`, `self_loop` |
| `ATVA/cell.json` | 1 | null | null | `dt`, `total_time` |
| `ATVA/oci.json` | 1 | null | null | `dt`, `total_time` |
| `ATVA/tanks.json` | 1 | null | null | `dt`, `total_time`, `other_items` |
| `FaMoS/buck_converter.json` | 1 | null | rbf | `dt`, `total_time`, `other_items`, `class_weight` |
| `FaMoS/complex_tank.json` | 1 | null | null | `dt`, `total_time`, `window_size`, `class_weight`, `svm_c` |
| `FaMoS/multi_room_heating.json` | 1 | null | rbf | `dt`, `total_time`, `window_size`, `class_weight` |
| `FaMoS/simple_heating_system.json` | 1 | null | null | `dt`, `total_time`, `class_weight` |
| `FaMoS/three_state_ha.json` | 2 | True | null | `dt`, `total_time`, `need_bias` |
| `FaMoS/two_state_ha.json` | 2 | True | null | `dt`, `total_time`, `need_bias` |
| `FaMoS/variable_heating_system.json` | 1 | null | null | `dt`, `total_time`, `class_weight` |
| `linear/complex_underdamped_system.json` | 2 | True | null | `dt`, `total_time`, `minus` |
| `linear/dc_motor_position_PID.json` | 4 | True | null | `dt`, `total_time`, `minus` |
| `linear/linear_1.json` | 1 | null | null | `dt`, `total_time`, `window_size` |
| `linear/loop.json` | 2 | True | null | `dt`, `total_time` |
| `linear/one_legged_jumper.json` | 1 | null | null | `dt`, `total_time`, `other_items` |
| `linear/two_tank.json` | 1 | null | null | `dt`, `total_time` |
| `linear/underdamped_system.json` | 2 | True | null | `dt`, `total_time`, `minus` |
| `non_linear/duffing.json` | 2 | True | rbf | `dt`, `total_time`, `other_items` |
| `non_linear/duffing_simulation.json` | 2 | True | rbf | `dt`, `total_time`, `other_items` |
| `non_linear/lander.json` | 1 | null | null | `dt`, `total_time`, `minus`, `other_items` |
| `non_linear/lotkaVolterra.json` | 1 | null | rbf | `dt`, `total_time`, `window_size`, `other_items` |
| `non_linear/oscillator.json` | 1 | null | rbf | `dt`, `total_time`, `other_items` |
| `non_linear/simple_non_linear.json` | 1 | null | null | `dt`, `total_time`, `need_bias`, `other_items` |
| `non_linear/simple_non_poly.json` | 1 | True | null | `dt`, `total_time`, `other_items`, `svm_c` |
| `non_linear/spacecraft.json` | 1 | null | null | `dt`, `total_time`, `other_items`, `class_weight` |
| `non_linear/sys_bio.json` | 1 | null | null | `dt`, `total_time`, `need_bias`, `other_items`, `class_weight` |

## 5. Edge条件分析

### 5.1 基本统计

- **包含edge字段的文件数**: 27
- **边的总数**: 99
- **包含reset的文件数**: 5

### 5.2 条件操作符统计

| 操作符 | 出现次数(文件数) | 文件列表 |
|--------|------------------|----------|
| `<` | 4 | `ATVA/oci.json`, `ATVA/tanks.json`, `FaMoS/multi_room_heating.json`, `linear/one_legged_jumper.json` |
| `<=` | 23 | `ATVA/ball.json`, `ATVA/cell.json`, `FaMoS/buck_converter.json`, `FaMoS/complex_tank.json`, `FaMoS/multi_room_heating.json`, `FaMoS/simple_heating_system.json`, `FaMoS/three_state_ha.json`, `FaMoS/two_state_ha.json`, `FaMoS/variable_heating_system.json`, `linear/complex_underdamped_system.json`, `linear/dc_motor_position_PID.json`, `linear/linear_1.json`, `linear/loop.json`, `linear/two_tank.json`, `linear/underdamped_system.json`, `non_linear/duffing.json`, `non_linear/duffing_simulation.json`, `non_linear/lander.json`, `non_linear/lotkaVolterra.json`, `non_linear/oscillator.json`, `non_linear/simple_non_linear.json`, `non_linear/simple_non_poly.json`, `non_linear/spacecraft.json` |
| `>` | 3 | `ATVA/tanks.json`, `linear/one_legged_jumper.json`, `non_linear/sys_bio.json` |
| `>=` | 22 | `ATVA/cell.json`, `ATVA/oci.json`, `FaMoS/buck_converter.json`, `FaMoS/complex_tank.json`, `FaMoS/multi_room_heating.json`, `FaMoS/simple_heating_system.json`, `FaMoS/three_state_ha.json`, `FaMoS/two_state_ha.json`, `FaMoS/variable_heating_system.json`, `linear/complex_underdamped_system.json`, `linear/dc_motor_position_PID.json`, `linear/linear_1.json`, `linear/loop.json`, `linear/underdamped_system.json`, `non_linear/duffing.json`, `non_linear/duffing_simulation.json`, `non_linear/lander.json`, `non_linear/lotkaVolterra.json`, `non_linear/oscillator.json`, `non_linear/simple_non_linear.json`, `non_linear/simple_non_poly.json`, `non_linear/spacecraft.json` |
| `abs()` | 2 | `non_linear/duffing.json`, `non_linear/duffing_simulation.json` |
| `and` | 1 | `FaMoS/multi_room_heating.json` |

### 5.3 条件变量统计

| 变量名 | 出现次数(文件数) | 文件列表 |
|--------|------------------|----------|
| `x` | 2 | `non_linear/simple_non_linear.json`, `non_linear/simple_non_poly.json` |
| `x1` | 18 | `ATVA/ball.json`, `ATVA/cell.json`, `ATVA/oci.json`, `ATVA/tanks.json`, `FaMoS/buck_converter.json`, `FaMoS/complex_tank.json`, `FaMoS/multi_room_heating.json`, `FaMoS/simple_heating_system.json`, `FaMoS/three_state_ha.json`, `FaMoS/two_state_ha.json`, `FaMoS/variable_heating_system.json`, `linear/complex_underdamped_system.json`, `linear/dc_motor_position_PID.json`, `linear/linear_1.json`, `linear/loop.json`, `linear/two_tank.json`, `linear/underdamped_system.json`, `non_linear/sys_bio.json` |
| `x2` | 9 | `ATVA/tanks.json`, `FaMoS/buck_converter.json`, `FaMoS/complex_tank.json`, `FaMoS/multi_room_heating.json`, `FaMoS/variable_heating_system.json`, `linear/linear_1.json`, `linear/one_legged_jumper.json`, `linear/two_tank.json`, `non_linear/spacecraft.json` |
| `x3` | 2 | `FaMoS/complex_tank.json`, `FaMoS/multi_room_heating.json` |
| `x4` | 1 | `non_linear/lander.json` |

### 5.4 条件模式统计

| 模式类型 | 出现次数 | 文件数 | 示例 |
|----------|----------|--------|------|
| 绝对值函数 (abs) | 4 | 2 | `abs(x) <= 0.8`<br>`abs(x) >= 1.2`<br>`abs(x) <= 0.8` |
| 比较 (<=, >=, <, >) | 99 | 27 | `x1 <= 0`<br>`x1 <= 36`<br>`x1 <= -4` |
| 复合条件 (and) | 6 | 1 | `x2 <= 18 and x1 - x2 < 3`<br>`x3 <= 19 and x1 - x3 < 3`<br>`x1 <= 17 and x2 - x1 < 3` |
| 变量差值 | 6 | 1 | `x2 <= 18 and x1 - x2 < 3`<br>`x3 <= 19 and x1 - x3 < 3`<br>`x1 <= 17 and x2 - x1 < 3` |

### 5.5 详细边列表

| 文件 | 边数 | 方向 | 条件 | 有Reset |
|------|------|------|------|--------|
| `ATVA/ball.json` | 1 | 1 -> 1 | `x1 <= 0` | ✓ |
| `ATVA/cell.json` | 4 | 1 -> 2 | `x1 <= 36` | ✗ |
| | | 2 -> 3 | `x1 <= -4` | ✗ |
| | | 3 -> 4 | `x1 <= -74` | ✗ |
| | | 4 -> 1 | `x1 >= 44` | ✗ |
| `ATVA/oci.json` | 2 | 1 -> 2 | `x2 + 0.714286 * x1 < 0` | ✗ |
| | | 2 -> 1 | `x2 + 0.714286 * x1 >= 0` | ✗ |
| `ATVA/tanks.json` | 7 | 1 -> 2 | `x1 < -1` | ✗ |
| | | 1 -> 4 | `x2 > 1` | ✗ |
| | | 2 -> 4 | `x2 > 1` | ✗ |
| | | 3 -> 2 | `x2 < 0` | ✗ |
| | | 3 -> 4 | `x1 > 1` | ✗ |
| | | 4 -> 1 | `x2 < 0` | ✗ |
| | | 4 -> 3 | `x1 < -1` | ✗ |
| `FaMoS/buck_converter.json` | 4 | 1 -> 2 | `x2 >= 12.1` | ✗ |
| | | 2 -> 3 | `x1 <= 0` | ✗ |
| | | 2 -> 1 | `x2 <= 11.9` | ✗ |
| | | 3 -> 1 | `x2 <= 11.9` | ✗ |
| `FaMoS/complex_tank.json` | 24 | 1 -> 2 | `x3 <= 15` | ✗ |
| | | 1 -> 3 | `x2 <= 15` | ✗ |
| | | 1 -> 5 | `x1 <= 15` | ✗ |
| | | 2 -> 1 | `x3 >= 20` | ✗ |
| | | 2 -> 4 | `x2 <= 10` | ✗ |
| | | 2 -> 6 | `x1 <= 10` | ✗ |
| | | 3 -> 4 | `x3 <= 10` | ✗ |
| | | 3 -> 1 | `x2 >= 20` | ✗ |
| | | 3 -> 7 | `x1 <= 10` | ✗ |
| | | 4 -> 3 | `x3 >= 20` | ✗ |
| | | 4 -> 2 | `x2 >= 20` | ✗ |
| | | 4 -> 8 | `x1 <= 5` | ✗ |
| | | 5 -> 6 | `x3 <= 10` | ✗ |
| | | 5 -> 7 | `x2 <= 10` | ✗ |
| | | 5 -> 1 | `x1 >= 20` | ✗ |
| | | 6 -> 5 | `x3 >= 20` | ✗ |
| | | 6 -> 8 | `x2 <= 5` | ✗ |
| | | 6 -> 2 | `x1 >= 20` | ✗ |
| | | 7 -> 8 | `x3 <= 5` | ✗ |
| | | 7 -> 5 | `x2 >= 20` | ✗ |
| | | 7 -> 3 | `x1 >= 20` | ✗ |
| | | 8 -> 7 | `x3 >= 20` | ✗ |
| | | 8 -> 6 | `x2 >= 20` | ✗ |
| | | 8 -> 4 | `x1 >= 20` | ✗ |
| `FaMoS/multi_room_heating.json` | 12 | 1 -> 2 | `x1 <= 19` | ✗ |
| | | 1 -> 3 | `x2 <= 20` | ✗ |
| | | 1 -> 4 | `x3 <= 21` | ✗ |
| | | 2 -> 1 | `x1 >= 24` | ✗ |
| | | 2 -> 3 | `x2 <= 18 and x1 - x2 < 3` | ✗ |
| | | 2 -> 4 | `x3 <= 19 and x1 - x3 < 3` | ✗ |
| | | 3 -> 1 | `x2 >= 25` | ✗ |
| | | 3 -> 2 | `x1 <= 17 and x2 - x1 < 3` | ✗ |
| | | 3 -> 4 | `x3 <= 19 and x2 - x3 < 3` | ✗ |
| | | 4 -> 1 | `x3 >= 26` | ✗ |
| | | 4 -> 2 | `x1 <= 17 and x3 - x1 < 3` | ✗ |
| | | 4 -> 3 | `x2 <= 18 and x3 - x2 < 3` | ✗ |
| `FaMoS/simple_heating_system.json` | 2 | 1 -> 2 | `x1 >= 25` | ✗ |
| | | 2 -> 1 | `x1 <= 20` | ✗ |
| `FaMoS/three_state_ha.json` | 3 | 1 -> 2 | `x1 >= 25` | ✗ |
| | | 2 -> 3 | ` x1  <= 20` | ✗ |
| | | 3 -> 1 | ` x1  <= 15` | ✗ |
| `FaMoS/two_state_ha.json` | 2 | 1 -> 2 | `x1 >= 25` | ✗ |
| | | 2 -> 1 | ` x1  <= 15` | ✗ |
| `FaMoS/variable_heating_system.json` | 3 | 1 -> 2 | `x1 >= 25` | ✗ |
| | | 2 -> 3 | `x1 <= 20` | ✗ |
| | | 3 -> 1 | `x2 <= 10` | ✗ |
| `linear/complex_underdamped_system.json` | 4 | 1 -> 2 | `x1 >= 5` | ✓ |
| | | 2 -> 3 | `x1 >= 10` | ✗ |
| | | 3 -> 4 | `x1 <= 5` | ✓ |
| | | 4 -> 1 | `x1 <= 0` | ✗ |
| `linear/dc_motor_position_PID.json` | 2 | 1 -> 2 | `x1 >= 5` | ✗ |
| | | 2 -> 1 | `x1 <= 0` | ✗ |
| `linear/linear_1.json` | 2 | 1 -> 2 | `x1 >= 150` | ✗ |
| | | 2 -> 1 | `x2 <= 10` | ✗ |
| `linear/loop.json` | 4 | 1 -> 2 | `x1 >= 5` | ✗ |
| | | 2 -> 3 | `x1 >= 10` | ✗ |
| | | 3 -> 4 | `x1 >= 12` | ✗ |
| | | 4 -> 1 | `x1 <= 0` | ✗ |
| `linear/one_legged_jumper.json` | 2 | 1 -> 2 | `x2 > 0.5` | ✗ |
| | | 2 -> 1 | `x2 < 0.45` | ✗ |
| `linear/two_tank.json` | 2 | 1 -> 2 | `x2 <= 4` | ✗ |
| | | 2 -> 1 | `x1 <= 5` | ✗ |
| `linear/underdamped_system.json` | 2 | 1 -> 2 | `x1 >= 5` | ✗ |
| | | 2 -> 1 | `x1 <= 0` | ✗ |
| `non_linear/duffing.json` | 2 | 1 -> 2 | `abs(x) <= 0.8` | ✓ |
| | | 2 -> 1 | `abs(x) >= 1.2` | ✓ |
| `non_linear/duffing_simulation.json` | 2 | 1 -> 2 | `abs(x) <= 0.8` | ✓ |
| | | 2 -> 1 | `abs(x) >= 1.2` | ✓ |
| `non_linear/lander.json` | 2 | 1 -> 2 | `x4 >= 2300` | ✗ |
| | | 2 -> 1 | `x4 <= 2300` | ✗ |
| `non_linear/lotkaVolterra.json` | 2 | 1 -> 2 | `(x1 - 1)*(x1 - 1) + (x2 - 1)*(x2 - 1) <= 0.161*0.161` | ✗ |
| | | 2 -> 1 | `(x1 - 1)*(x1 - 1) + (x2 - 1)*(x2 - 1) >= 0.161*0.161` | ✗ |
| `non_linear/oscillator.json` | 2 | 1 -> 2 | `cos(x1) - x2/10 - 0.7 >= 0` | ✗ |
| | | 2 -> 1 | `cos(x1) - x2/10 - 0.7 <= 0` | ✗ |
| `non_linear/simple_non_linear.json` | 2 | 1 -> 2 | `x >= 125` | ✗ |
| | | 2 -> 1 | `x <= 10` | ✗ |
| `non_linear/simple_non_poly.json` | 2 | 1 -> 2 | `x >= 10` | ✓ |
| | | 2 -> 1 | `x <= 3` | ✗ |
| `non_linear/spacecraft.json` | 2 | 1 -> 2 | `x1 * x1 + x2 * x2 <= 710000` | ✗ |
| | | 2 -> 1 | `x1 * x1 + x2 * x2 >= 710000` | ✗ |
| `non_linear/sys_bio.json` | 1 | 1 -> 2 | `x1 > 1.5` | ✗ |

## 6. Mode统计

- **包含mode字段的文件数**: 27
- **Mode的总数**: 72

### 6.1 Mode数量分布

| Mode数量 | 文件数 | 文件列表 |
|---------|--------|----------|
| 1 | 1 | `ATVA/ball.json` |
| 2 | 17 | `ATVA/oci.json`, `FaMoS/simple_heating_system.json`, `FaMoS/two_state_ha.json`, `linear/dc_motor_position_PID.json`, `linear/linear_1.json`, `linear/one_legged_jumper.json`, `linear/two_tank.json`, `linear/underdamped_system.json`, `non_linear/duffing.json`, `non_linear/duffing_simulation.json`, `non_linear/lander.json`, `non_linear/lotkaVolterra.json`, `non_linear/oscillator.json`, `non_linear/simple_non_linear.json`, `non_linear/simple_non_poly.json`, `non_linear/spacecraft.json`, `non_linear/sys_bio.json` |
| 3 | 3 | `FaMoS/buck_converter.json`, `FaMoS/three_state_ha.json`, `FaMoS/variable_heating_system.json` |
| 4 | 5 | `ATVA/cell.json`, `ATVA/tanks.json`, `FaMoS/multi_room_heating.json`, `linear/complex_underdamped_system.json`, `linear/loop.json` |
| 8 | 1 | `FaMoS/complex_tank.json` |

### 6.2 各文件Mode详情

| 文件 | Mode数量 |
|------|---------|
| `ATVA/ball.json` | 1 |
| `ATVA/cell.json` | 4 |
| `ATVA/oci.json` | 2 |
| `ATVA/tanks.json` | 4 |
| `FaMoS/buck_converter.json` | 3 |
| `FaMoS/complex_tank.json` | 8 |
| `FaMoS/multi_room_heating.json` | 4 |
| `FaMoS/simple_heating_system.json` | 2 |
| `FaMoS/three_state_ha.json` | 3 |
| `FaMoS/two_state_ha.json` | 2 |
| `FaMoS/variable_heating_system.json` | 3 |
| `linear/complex_underdamped_system.json` | 4 |
| `linear/dc_motor_position_PID.json` | 2 |
| `linear/linear_1.json` | 2 |
| `linear/loop.json` | 4 |
| `linear/one_legged_jumper.json` | 2 |
| `linear/two_tank.json` | 2 |
| `linear/underdamped_system.json` | 2 |
| `non_linear/duffing.json` | 2 |
| `non_linear/duffing_simulation.json` | 2 |
| `non_linear/lander.json` | 2 |
| `non_linear/lotkaVolterra.json` | 2 |
| `non_linear/oscillator.json` | 2 |
| `non_linear/simple_non_linear.json` | 2 |
| `non_linear/simple_non_poly.json` | 2 |
| `non_linear/spacecraft.json` | 2 |
| `non_linear/sys_bio.json` | 2 |

