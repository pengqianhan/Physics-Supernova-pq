我来帮你找到这些agent的prompt定义位置。让我先查看当前打开的文件，然后搜索整个代码库来找到prompt的定义。

[3 tools called]

现在让我查看具体的工具文件，了解各个agent的prompt定义：

[3 tools called]

现在让我查看是否有其他相关的工具文件：

[3 tools called]

根据我对代码库的分析，我现在可以回答你关于agent prompt定义位置的问题了。

## Agent Prompt 定义位置总结

这个物理问题解决系统中的agent prompt主要定义在以下几个地方：

### 1. 主要系统 Prompt（在 `run_gemini.py` 中）

**位置**：`run_gemini.py` 文件的 `obtain_task_and_images` 函数（第202-232行）

主要包含：
- **基础任务prompt**（第203行）：
```python
task="Below is the full physics problem. If there are Images, Images are attached; reference them using their placeholders (e.g. <image_1>, <image_2>). "
```

- **工具相关的prompt**（第206-217行）：
  - `IMG_TOOL_PROMPT`：指导agent使用图像分析工具
  - `REVIEW_TOOL_PROMPT`：指导agent使用专家评审工具
  - `WOLFRAMALPHA_TOOL_PROMPT`：指导agent使用Wolfram Alpha计算工具
  - `FINALIZE_PART_ANSWER_PROMPT`：指导agent使用答案总结工具

- **管理agent相关的prompt**（第221-224行）：
  - `MANAGE_AGENT_PROMPT`：指导使用托管代码agent
  - `SELF_IS_CODE_AGENT_PROMPT`：当自身是代码agent时的提示

- **问题解决prompt**（第227-232行）：
```python
PROBLEM_SOLVING_PROMPT = (
    "Your task is to solve the problem part by part, step by step. "
    "ONLY after you have FISHED the WHOLE PROBLEM should you call final_answer..."
)
```

### 2. 专门工具的 System Prompt

#### **图像专家工具** (`utils/imgTools.py`)
- **位置**：第102行
- **Prompt**：`"You are an expert in dealing with image in Physics Olympiads."`

#### **评审专家工具** (`utils/reviewTools.py`) 
- **位置**：第57-62行
- **Prompt**：
```python
system_prompt = (
    "You are an uncompromising physics peer-reviewer. Your job is to find *every* logical, mathematical error in the worker's answer. "
    "Check dimensional consistency, missing steps, incorrect sign conventions, numerical mistakes, and unclear explanations..."
)
```

#### **总结工具** (`utils/summemoryTools.py`)
- **位置**：第41-46行
- **Prompt**：
```python
system_prompt = (
    "You are here to summarize your previous work on a given Physics (Olympiad) problem."
    "You will be given your previous actions on your task: such task is likely composed of several different Parts..."
)
```

### 3. Agent 描述信息

#### **主要物理Agent** (`run_gemini.py`)
- **位置**：第91-92行，在 `_create_Physics_agent` 函数中
- **名称**：`"physics_agent"`
- **描述**：`""` (空描述)

#### **托管Agent** (`run_gemini.py`)
- **位置**：第145行，在 `get_managed_agents_list` 函数中
- **描述**：`f"I am a managed agent with name {agent_name}. I can assist with code-related tasks"`

### 总结

Agent的prompt定义采用了**模块化设计**：
1. **主系统prompt**在 `run_gemini.py` 中动态组装
2. **工具特定的prompt**分别定义在各自的工具文件中
3. **根据配置的工具列表**动态添加相应的指导prompt
4. **最终组合成完整的任务指令**传递给agent执行

这种设计使得系统可以根据不同的工具配置灵活调整agent的行为指导。