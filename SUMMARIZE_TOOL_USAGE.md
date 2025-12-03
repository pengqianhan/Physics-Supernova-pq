# HA迭代总结工具使用说明

## 概述

`SummarizeMemoryTool` (工具名: `summarize_ha_iterations`) 是一个专门用于Hybrid Automaton学习迭代过程总结的工具。该工具能够从agent的memory中提取完整的HA规范迭代历史，从v0（初始规范）到最新版本。

## 功能特性

该工具会返回：
1. **所有HA版本** - 从v0到最新版本的完整历史
2. **完整的Python dict规范** - 每个版本的完整HA规范（包括automaton和config部分）
3. **分析/推理** - 每个版本的改进原因：
   - v0: 初始理解或假设
   - v1+: 前一版本发现的问题、指标反馈、所做的改进
4. **评估指标** - tc（change-point error）、max_diff、mean_diff等
5. **审查反馈** - 如果使用了review tool，包含专家审查意见

### 可选功能：Memory重置（用于遗传算法）

工具支持可选的 `reset_memory` 参数：
- `reset_memory=False`（默认）：仅总结，保留完整的agent memory，可以继续基于历史上下文进行refinement
- `reset_memory=True`：总结后重置agent memory，适用于遗传算法场景，每代agent从干净状态开始新迭代

## 输出格式

工具返回的总结采用以下Markdown格式：

```markdown
## Complete HA Learning Iteration History

## Version v0 (Initial Specification)
### HA Specification:
```python
{
  "automaton": {...},
  "config": {...}
}
```
### Analysis:
[初始理解、假设或baseline方法]
### Metrics:
[如果已评估则显示指标，否则显示"Not evaluated yet"]
### Review Feedback:
[如果已审查则显示反馈，否则显示"Not reviewed yet"]

## Version v1
### HA Specification:
```python
{
  "automaton": {...},
  "config": {...}
}
```
### Analysis:
[v0中发现的问题、指标反馈、所做的改进]
### Metrics:
[评估指标]
### Review Feedback:
[专家审查反馈]

## Version v2
...
```

## 使用方法

### 1. 在run_llm_ha_beta.py中使用

工具已集成到 `run_llm_ha_beta.py` 中，默认包含在工具列表中：

```bash
python run_llm_ha_beta.py \
  --input-data-path utils/Dainarx_code/data_duffing \
  --manager-type CodeAgent \
  --tools-list hybrid_automaton_image_analysis ask_review_expert_ha summarize_ha_iterations
```

### 2. 配置总结工具使用的模型

可以通过命令行参数指定总结工具使用的LLM模型：

```bash
python run_llm_ha_beta.py \
  --input-data-path utils/Dainarx_code/data_duffing \
  --summarize-tool-model "gemini/gemini-2.5-flash-lite"
```

默认模型：`gemini/gemini-2.5-flash-lite`

### 3. Agent调用方式

在agent运行过程中，当完成多次HA规范迭代后，可以调用该工具：

**标准调用（仅总结，保留memory）：**
```python
# 默认行为：总结所有迭代，memory保持不变
result = agent.tools['summarize_ha_iterations'].forward()
# 或显式指定
result = agent.tools['summarize_ha_iterations'].forward(reset_memory=False)
```

**遗传算法场景（总结并重置memory）：**
```python
# 适用于遗传算法：总结当前代的迭代历史，然后重置memory开始新一代
result = agent.tools['summarize_ha_iterations'].forward(reset_memory=True)
```

**工具会自动：**
- 从agent的memory中提取所有HA迭代历史
- 生成结构化的总结
- 如果 `reset_memory=True`，清空memory并准备新的干净状态

## 工具配置

### 初始化参数

- `worker_agent`: 主agent的引用（用于访问memory）
- `summarize_model_id`: 总结模型ID，默认 `"gemini/gemini-2.5-flash-lite"`

### 环境变量

确保设置了以下环境变量：
```bash
GEMINI_API_KEY=your_api_key_here
```

## 修改内容

### 1. utils/summemoryTools_ha.py
- 更新系统提示词：从物理问题总结改为HA规范迭代总结
- 修改输出格式：要求包含完整的HA dict及分析
- 工具名改为 `summarize_ha_iterations`
- API配置改为使用Gemini API

### 2. run_llm_ha_beta.py
- 导入 `SummarizeMemoryTool`
- 添加到 `TOOLNAME2TOOL` 映射
- 添加工具初始化逻辑（支持自定义模型）
- 添加命令行参数 `--summarize-tool-model`
- 将工具添加到默认工具列表

## 适用场景

### 标准使用场景（reset_memory=False，默认）

1. **完成多次迭代后** - 想要查看从v0到最新版本的完整演化过程
2. **文档记录** - 需要记录整个HA学习过程的详细文档
3. **分析改进模式** - 识别哪些类型的改进最有效
4. **继续优化** - 基于历史上下文继续refine HA规范

### 遗传算法场景（reset_memory=True）

5. **种群进化** - 每个agent代表种群中的一个个体，完成一代进化后：
   - 总结该个体在当前代的所有HA版本
   - 重置memory，准备下一代的fresh start
   - 保留总结结果用于fitness评估和选择

6. **多代进化** - 在遗传算法的代际循环中：
   ```python
   for generation in range(num_generations):
       # Agent在当前代进行HA refinement迭代
       agent.run(task, images=images)

       # 总结当前代的所有迭代，并重置memory准备下一代
       summary = agent.tools['summarize_ha_iterations'].forward(reset_memory=True)

       # 从总结中提取最终HA规范用于fitness评估
       final_ha = extract_final_version(summary)
       fitness = evaluate_fitness(final_ha)

       # 基于fitness进行选择、交叉、变异等遗传操作
       # ...
   ```

7. **岛屿模型（Island Model）** - 多个隔离种群并行进化：
   - 每个岛屿的agent独立进化
   - 定期总结并重置，在岛屿间迁移优秀个体
   - 避免memory污染导致的种群同质化

## 注意事项

1. **完整性保证** - 工具会提取所有版本，不会遗漏任何中间版本
2. **格式要求** - 返回的dict格式符合Python/JSON规范，可直接解析
3. **上下文长度** - 使用16384 tokens的输出限制，足够支持多版本总结
4. **重试机制** - 内置3次重试逻辑，确保总结生成的鲁棒性

### Memory重置行为说明

5. **reset_memory=False（默认）**：
   - Agent保留完整的conversation history
   - 可以继续基于所有历史context进行refinement
   - 适合单agent连续优化场景

6. **reset_memory=True（遗传算法）**：
   - Agent memory被清空，只保留初始task
   - 初始task会被更新，提示agent这是新一代
   - 总结结果需要在外部保存（用于fitness评估、选择等）
   - 适合需要多代独立进化的场景

7. **首次使用标记**：
   - `is_first_use` 标志确保task更新只在第一次reset时发生
   - 后续reset不会重复添加提示信息

## 示例输出

agent完成3次迭代后调用工具，会得到：

```markdown
## Complete HA Learning Iteration History

## Version v0 (Initial Specification)
### HA Specification:
```python
{
  "automaton": {
    "var": "x1, x2",
    "input": "u1",
    "mode": [{"id": 1, "eq": "x1[1] = x2[0], x2[1] = -x1[0] + u1"}],
    "edge": []
  },
  "config": {"dt": 0.001, "total_time": 10.0}
}
```
### Analysis:
Initial linear approximation of the Duffing oscillator
### Metrics:
tc: 0.0, max_diff: 2.45, mean_diff: 0.83
### Review Feedback:
Missing cubic nonlinearity term in x2 equation

## Version v1
### HA Specification:
```python
{
  "automaton": {
    "var": "x1, x2",
    "input": "u1",
    "mode": [{"id": 1, "eq": "x1[1] = x2[0], x2[1] = -x1[0] - x1[0]**3 + u1"}],
    "edge": []
  },
  "config": {"dt": 0.001, "total_time": 10.0}
}
```
### Analysis:
Added cubic term x1[0]**3 based on reviewer feedback. Missing damping term.
### Metrics:
tc: 0.0, max_diff: 1.12, mean_diff: 0.34
### Review Feedback:
Improvement in trajectory matching, but damping coefficient missing

## Version v2
...
```

## 遗传算法集成示例

以下是一个完整的遗传算法框架示例，展示如何使用 `reset_memory=True` 功能：

```python
import json
from run_llm_ha_beta import create_agent, obtain_task_and_images

class HAGeneticAlgorithm:
    def __init__(self, population_size=10, num_generations=20, num_islands=3):
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_islands = num_islands
        self.populations = [[] for _ in range(num_islands)]  # 岛屿模型

    def create_individual(self, island_id):
        """创建一个agent个体"""
        agent = create_agent(
            model_id="gemini/gemini-2.5-flash-lite",
            input_data_path="utils/Dainarx_code/data_duffing",
            tools_list=["hybrid_automaton_image_analysis",
                       "ask_review_expert_ha",
                       "summarize_ha_iterations"],
            manager_type="CodeAgent",
        )
        return agent

    def run_agent_iteration(self, agent, task, images, max_iterations=5):
        """运行agent的一次完整迭代（可能包含多个HA版本refinement）"""
        agent.run(task, images=images)

        # 总结该代的所有HA迭代，并重置memory准备下一代
        summary = agent.tools['summarize_ha_iterations'].forward(reset_memory=True)

        return summary

    def extract_final_ha_from_summary(self, summary):
        """从总结中提取最终（最新）版本的HA规范"""
        # 解析summary，提取最后一个版本
        # 这里需要实现具体的解析逻辑
        # 返回格式: {"automaton": {...}, "config": {...}}
        pass

    def evaluate_fitness(self, ha_dict, npz_file):
        """评估HA规范的fitness（基于metrics）"""
        from utils.Dainarx_code.HA_evaluation import HAEvaluator

        evaluator = HAEvaluator(
            ha_dict=ha_dict,
            npz_file_path=npz_file,
            dt=0.001,
            total_time=10.0
        )

        metrics_text, _ = evaluator(
            plot_mode="overlay",
            save_path=None,
            print_metrics=False
        )

        # 解析metrics并计算fitness
        # fitness = f(tc, max_diff, mean_diff)
        # 例如: fitness = 1 / (1 + tc + max_diff + mean_diff)
        pass

    def evolve_generation(self, island_id, generation):
        """进化一代"""
        task, images = obtain_task_and_images(
            input_data_path="utils/Dainarx_code/data_duffing",
            system_name="Duffing Oscillator",
            num_variables=1,
            num_inputs=1,
            tools_list=["hybrid_automaton_image_analysis",
                       "ask_review_expert_ha",
                       "summarize_ha_iterations"],
            manager_type="CodeAgent",
        )

        population = self.populations[island_id]
        new_population = []

        for individual in population:
            # 运行agent迭代
            summary = self.run_agent_iteration(individual, task, images)

            # 提取最终HA规范
            final_ha = self.extract_final_ha_from_summary(summary)

            # 评估fitness
            fitness = self.evaluate_fitness(
                final_ha,
                "utils/Dainarx_code/data_duffing/test_data0.npz"
            )

            new_population.append({
                'agent': individual,
                'ha_spec': final_ha,
                'fitness': fitness,
                'summary': summary
            })

        # 选择、交叉、变异
        new_population = self.selection(new_population)
        new_population = self.crossover(new_population)
        new_population = self.mutation(new_population)

        self.populations[island_id] = new_population

        return new_population

    def migrate_between_islands(self, generation):
        """岛屿间迁移优秀个体"""
        if generation % 5 == 0:  # 每5代迁移一次
            for i in range(self.num_islands):
                source_island = self.populations[i]
                target_island = self.populations[(i + 1) % self.num_islands]

                # 从source选择最优个体迁移到target
                best_individual = max(source_island, key=lambda x: x['fitness'])
                target_island.append(best_individual)

    def run(self):
        """运行完整的遗传算法"""
        # 初始化种群
        for island_id in range(self.num_islands):
            for _ in range(self.population_size):
                agent = self.create_individual(island_id)
                self.populations[island_id].append({'agent': agent})

        # 多代进化
        for generation in range(self.num_generations):
            print(f"\n=== Generation {generation} ===")

            # 每个岛屿并行进化
            for island_id in range(self.num_islands):
                print(f"Evolving Island {island_id}...")
                self.evolve_generation(island_id, generation)

            # 岛屿间迁移
            self.migrate_between_islands(generation)

            # 记录最优个体
            all_individuals = [ind for pop in self.populations for ind in pop]
            best = max(all_individuals, key=lambda x: x['fitness'])
            print(f"Best fitness: {best['fitness']}")

            # 保存最优HA规范
            with open(f"best_ha_gen_{generation}.json", "w") as f:
                json.dump(best['ha_spec'], f, indent=2)

        return best

# 使用示例
if __name__ == "__main__":
    ga = HAGeneticAlgorithm(
        population_size=10,
        num_generations=20,
        num_islands=3
    )
    best_solution = ga.run()
    print(f"Final best HA specification: {best_solution['ha_spec']}")
```

### 关键要点

1. **Memory管理**：每次调用 `summarize_ha_iterations(reset_memory=True)` 后，agent的memory被清空，可以开始全新的迭代
2. **总结保存**：总结结果包含完整的迭代历史，需要在外部保存用于fitness评估
3. **岛屿模型**：多个种群隔离进化，避免premature convergence
4. **迁移策略**：定期在岛屿间迁移优秀个体，保持种群多样性

## 相关文件

- `utils/summemoryTools_ha.py` - 工具实现
- `run_llm_ha_beta.py` - 主运行脚本
- `utils/imgTools_ha.py` - 图像分析工具
- `utils/reviewTools_ha.py` - 审查工具
