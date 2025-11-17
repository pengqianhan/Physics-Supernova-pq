# run_gemini.py 工作流循环分析

## 🔄 工作流循环体现的关键代码

### 1. max_steps - 定义最大循环次数

**位置 1**: `run_gemini.py:53, 91, 96`
```python
def _create_Physics_agent(Tools_list:List[type[Tool]],
                         markdown_content: MarkdownMessage,
                         model_id: str = "openrouter/google/gemini-2.5-pro",
                         managed_agents_list: List[MultiStepAgent] = None,
                         max_steps:int=80,  # ← 第53行：定义参数默认值
                         **kwargs):

    # initialize the manager agent
    manager_agent_kwargs = dict(
        model=model,
        tools=tools,
        max_steps=max_steps,  # ← 第91行：传递给Agent配置
        verbosity_level=2,
        name="physics_agent",
        description="",
        managed_agents=managed_agents_list
    ) # max_steps=80，which means the agent can only run 80 steps  # ← 第96行：注释说明
```

**作用**:
- `max_steps=80` 意味着 Agent 最多执行 **80 个推理-行动步骤**
- 每个步骤包括：LLM 思考 → 决定行动（调用工具/执行代码/返回答案）→ 获取结果
- 达到 80 步后，即使任务未完成，循环也会强制终止

**位置 2**: `run_gemini.py:148` (managed agents)
```python
managed_agent = CodeAgent(
    tools=[],
    model=model,
    name=agent_name,
    additional_authorized_imports=[...],
    description=f"I am a managed agent with name {agent_name}...",
    max_steps=80,  # ← 管理的子 Agent 也有 80 步限制
    verbosity_level=2,
)
```

---

### 2. managerAgent.run() - 循环入口

**位置**: `run_gemini.py:350`
```python
def main():
    args = parse_args()

    # 创建 Agent
    managerAgent = create_agent(
        model_id=args.manager_model,
        input_markdown_file=args.input_markdown_file,
        tools_list=args.tools_list,
        ...
    )

    # 准备任务和图像
    task, compressed_problem_images = obtain_task_and_images(
        input_markdown_file=args.input_markdown_file,
        tools_list=args.tools_list,
        ...
    )

    # ← 第350行：启动工作流循环
    managerAgent.run(task, images=compressed_problem_images)
```

**作用**:
- `managerAgent.run()` 是启动循环的**唯一入口**
- 这个方法内部实现了完整的 **Agent 推理循环**（位于 smolagents 库中）

---

### 3. Agent 类型选择 - 决定循环行为

**位置**: `run_gemini.py:98-108`
```python
if kwargs["manager_type"] == "CodeAgent":
    manager_agent_kwargs["additional_authorized_imports"] = [
        "os", "sys", "time", "argparse", "pathlib",
        "matplotlib.pyplot", "numpy", "pandas"
    ]
    managerAgent = CodeAgent(**manager_agent_kwargs)  # ← CodeAgent 循环
elif kwargs["manager_type"] == "ToolCallingAgent":
    manager_agent_kwargs["max_tool_threads"] = 1
    managerAgent = ToolCallingAgent(**manager_agent_kwargs)  # ← ToolCallingAgent 循环
else:
    raise ValueError(f"Unknown manager type...")
```

**区别**:
- **CodeAgent**: 循环中可以执行 Python 代码 + 调用工具
- **ToolCallingAgent**: 循环中只能调用工具，不能执行代码

---

### 4. verbosity_level - 循环过程可见性

**位置**: `run_gemini.py:92, 149`
```python
manager_agent_kwargs = dict(
    model=model,
    tools=tools,
    max_steps=max_steps,
    verbosity_level=2,  # ← 控制循环过程输出详细程度
    name="physics_agent",
    ...
)
```

**作用**:
- `verbosity_level=0`: 静默模式，不输出循环过程
- `verbosity_level=1`: 输出关键步骤
- `verbosity_level=2`: **详细输出每一步的思考、行动、工具调用结果**

---

## 📊 完整的循环流程（由 smolagents 实现）

虽然循环逻辑在 `smolagents` 库内部，但我们可以从配置推断出循环结构：

```
managerAgent.run(task, images) 启动
    │
    ├─→ Step 1:
    │     LLM 生成 → 决定行动（工具/代码/答案）→ 执行 → 记录结果
    │
    ├─→ Step 2:
    │     LLM 生成 → 决定行动 → 执行 → 记录结果
    │
    ├─→ Step 3:
    │     ...
    │
    ├─→ ...
    │
    └─→ Step 80 (max_steps 达到)
          或
        Agent 调用 final_answer() 提前结束
```

### 每个 Step 的内部流程：

1. **LLM 推理**: 基于当前上下文（任务 + 历史记录）生成下一步计划
2. **行动决策**:
   - 调用工具（如 `ask_image_expert`, `ask_review_expert`）
   - 执行代码（仅 CodeAgent）
   - 返回最终答案（`final_answer`）
3. **执行行动**: 运行工具或代码
4. **更新记忆**: 将结果添加到 Agent 的 memory
5. **检查终止条件**:
   - 是否达到 `max_steps`？
   - 是否调用了 `final_answer`？
   - 是否发生错误？

---

## 🎯 关键配置总结

| 代码位置 | 配置项 | 作用 | 默认值 |
|---------|--------|------|--------|
| `run_gemini.py:53` | `max_steps` 参数 | 定义循环最大步数 | 80 |
| `run_gemini.py:91` | `max_steps` 传递 | 配置到 Agent | 80 |
| `run_gemini.py:92` | `verbosity_level` | 控制循环输出详细度 | 2 |
| `run_gemini.py:98-108` | Agent 类型 | 决定循环中可执行的操作类型 | CodeAgent/ToolCallingAgent |
| `run_gemini.py:350` | `managerAgent.run()` | 启动循环 | - |

---

## 💡 如何查看循环执行过程？

由于 `verbosity_level=2`，运行时你会看到类似的输出：

```
╭──────────────────────────────────────── New run ─────────────────────────────────────────╮
│                                                                                           │
│ Step 0 - Executing task:                                                                 │
│ Below is the full physics problem...                                                     │
│                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────── New step ──────────────────────────────────────────╮
│                                                                                           │
│ Step 1 - Thought: I need to analyze the first image to understand the problem.           │
│ Action: ask_image_expert(image_ref="<image_1>", question="Describe this galaxy image")   │
│                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────── Tool output ───────────────────────────────────────╮
│                                                                                           │
│ The image shows NGC 6946 galaxy with rotation curve data...                              │
│                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────── New step ──────────────────────────────────────────╮
│                                                                                           │
│ Step 2 - Thought: Based on the image analysis, I can now solve Part B...                 │
│ Action: ... (继续下一步)                                                                  │
│                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

---

## 🔍 循环终止条件

循环会在以下情况终止：

1. **正常终止**: Agent 调用 `final_answer()` 返回最终答案
2. **达到限制**: 执行了 `max_steps=80` 步
3. **错误终止**: 工具调用失败或代码执行出错（取决于错误处理配置）

对应代码中的提示（`run_gemini.py:231`）：
```python
PROBLEM_SOLVING_PROMPT = (
    "Your task is to solve the problem part by part, step by step. "
    "ONLY after you have FISHED the WHOLE PROBLEM should you call final_answer, "
    "never call final_answer when there are parts left! "
    "Or the program will shut down immediately, and you would have NO CHANCE to continue solving!\n\n"
    ...
)
```

这段提示词告诉 Agent：
- 不要过早调用 `final_answer`
- 一旦调用 `final_answer`，循环**立即终止**

---

## 📚 延伸：smolagents 库中的实际循环代码

虽然 `run_gemini.py` 没有直接实现循环，但在 `smolagents` 库的 Agent 类中，循环大致如下：

```python
# 伪代码（实际在 smolagents 库中）
class CodeAgent:
    def run(self, task, images=None):
        self.memory.reset()
        self.memory.add_step(task, images)

        for step in range(self.max_steps):  # ← 循环在这里
            # 1. LLM 生成下一步
            llm_output = self.model.generate(self.memory.to_messages())

            # 2. 解析行动
            action = self.parse_action(llm_output)

            # 3. 执行行动
            if action.type == "tool_call":
                result = self.execute_tool(action.tool_name, action.args)
            elif action.type == "code":
                result = self.execute_code(action.code)
            elif action.type == "final_answer":
                return action.answer  # ← 提前退出循环

            # 4. 更新记忆
            self.memory.add_step(action, result)

            # 5. 输出（如果 verbosity_level > 0）
            if self.verbosity_level >= 2:
                print(f"Step {step}: {action} → {result}")

        # 达到 max_steps
        raise MaxStepsReached(f"Reached max_steps={self.max_steps}")
```

---

## 总结

**run_gemini.py 中体现循环的关键点**:

1. ✅ **max_steps=80** (行 53, 91, 96, 148) - 定义循环次数
2. ✅ **verbosity_level=2** (行 92, 149) - 展示循环过程
3. ✅ **managerAgent.run()** (行 350) - 启动循环
4. ✅ **Agent 类型选择** (行 98-108) - 决定循环中的能力
5. ✅ **PROBLEM_SOLVING_PROMPT** (行 229-234) - 指导循环中的行为

**真正的循环实现** 在 `smolagents` 库的 `CodeAgent.run()` 或 `ToolCallingAgent.run()` 方法中。
