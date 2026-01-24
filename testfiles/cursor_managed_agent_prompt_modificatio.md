# Managed agent prompt modification
_Exported on 1/24/2026 at 17:20:41 GMT+13 from Cursor (2.4.21)_

---

**User**

使用Context7 来搜索官方文档，然后告诉我managed agent 的 prompt 如何修改呢？

---

**Cursor**

Let me first check the available tools from Context7 MCP server to search the official documentation.

Let me first resolve the library ID for smolagents, then query the documentation about managed agent prompts.

找到了 smolagents 的库 ID。现在让我查询关于 managed agent prompt 修改的文档。

根据 Context7 搜索到的官方文档，以下是修改 managed agent prompt 的方法：

## Managed Agent Prompt 修改方法

### 1. 通过 `name` 和 `description` 参数设置

当创建 managed agent 时，**必须设置 `name` 和 `description` 属性**，这些会被插入到 manager agent 的 system prompt 中：

```python
from smolagents import CodeAgent, InferenceClientModel

# 创建一个 managed agent
web_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="web_search_agent",  # agent 名称
    description="Runs web searches for you. Give it your query as an argument."  # agent 描述
)

# 将其加入 manager agent
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent]
)
```

Manager agent 的 system prompt 模板会自动使用这些属性：

```jinja
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
{%- endfor %}
```

### 2. 通过 `instructions` 参数添加自定义指令

可以在创建 agent 时传入 `instructions` 参数，这些指令会**追加**到 system prompt 末尾：

```python
agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    instructions="Always explain your reasoning step by step. Be concise."
)
```

### 3. 直接修改 `prompt_templates`

对于更复杂的定制，可以直接修改 agent 的 `prompt_templates` 字典：

```python
# 追加内容到现有的 system prompt
agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + "\nHere you go!"
```

你也可以**完全替换** system prompt 模板，但需要保留必要的占位符（如 `{{ tool.description }}` 和 managed agents 相关的占位符）。

### 4. 使用 `ManagedAgentPromptTemplate`

smolagents 还提供了专门的 `ManagedAgentPromptTemplate` 类：

```python
class ManagedAgentPromptTemplate(PromptTemplates):
    """Prompt template for managed agents."""
    def __init__(self):
        super().__init__(
            template='Agent instructions: {instructions}\nContext: {context}\nCommand: {command}'
        )
```

---

让我看一下你项目中现有的 managed agent 实现：

根据你的项目代码，我可以看到 managed agent 的 prompt 是通过以下几种方式设置的：

## 你项目中 Managed Agent Prompt 的修改方法

### 1. 修改 `description` 参数（主要方式）

在 `run_llm_ha_gamma.py` 第 280-299 行，`managed_agent_description` 变量定义了 agent 的描述：

```280:298:run_llm_ha_gamma.py
        managed_agent_description = f"""I am a managed agent with name {agent_name}. I can assist with code-related tasks.

## Available npz files
You have access to the following NPZ data files:
{npz_files_description}

**Quick Access via State Variables**
- `DATA_FILE_PATHS`: List of all available NPZ file paths
- `DATA_FILE_PATH`: Path to the first/primary data file (for convenience)

**Example Usage**
```python
import numpy as np
# Load a specific file
data = np.load(DATA_FILE_PATHS[0])
# Or use the primary file
data = np.load(DATA_FILE_PATH)
```

"""
```

然后在创建 `CodeAgent` 时传入：

```python
managed_agent = CodeAgent(
    ...
    description=managed_agent_description,  # 第 323 行
    ...
)
```

**这个 `description` 会被插入到 manager agent 的 system prompt 中**，manager agent 可以看到并根据此描述来决定如何调用这个 managed agent。

### 2. 修改 `instructions` 参数（为 managed agent 本身添加指令）

如果你想给 managed agent 本身添加额外的行为指令，可以在创建时传入 `instructions`：

```python
managed_agent = CodeAgent(
    tools=[],
    model=model,
    name=agent_name,
    description=managed_agent_description,
    instructions="Always provide step-by-step explanations. Double check your calculations.",  # 新增
    max_steps=80,
    ...
)
```

### 3. 直接修改 `prompt_templates`（高级用法）

创建 agent 后，可以直接修改其 prompt 模板：

```python
managed_agent = CodeAgent(...)

# 追加内容到 system prompt
managed_agent.prompt_templates["system_prompt"] += "\n\n## Additional Guidelines\n- Always validate input data..."

# 或者完全替换（需谨慎，保留必要占位符）
# managed_agent.prompt_templates["system_prompt"] = custom_prompt_template
```

### 实际修改示例

如果你想让 managed agent 有更具体的行为指引，可以这样修改 `get_managed_agents_list` 函数：

```python
managed_agent_description = f"""I am a data analysis expert for hybrid automaton learning.

## My Capabilities
- Analyze trajectory data from NPZ files
- Extract statistical features (mean, std, change points)
- Identify system modes and transitions

## Available NPZ Files
{npz_files_description}

## How to Use Me
Provide clear instructions about what analysis you need.
I will return structured results with my findings.
"""

# 给 agent 本身的额外指令
agent_instructions = """
When analyzing data:
1. Always check data shape and validity first
2. Use numpy for calculations (avoid Python built-ins like len(), list())
3. Return results as structured dictionaries
4. Handle edge cases gracefully
"""

managed_agent = CodeAgent(
    tools=[],
    model=model,
    name=agent_name,
    description=managed_agent_description,
    instructions=agent_instructions,  # 新增
    ...
)
```

总结一下：
- **`description`**: 给 manager agent 看的，描述这个 managed agent 能做什么
- **`instructions`**: 给 managed agent 自己看的，指导其行为方式
- **`prompt_templates`**: 更底层的模板定制

---

**User**

description: 给 manager agent 看的，描述这个 managed agent 能做什么
instructions: 给 managed agent 自己看的，指导其行为方式 这两句的出处是什么？给我出处的链接

---

**Cursor**

让我再次搜索 Context7 文档来找到准确的出处链接。

找到了官方文档的出处链接：

## 1. `description` - 给 manager agent 看的

**出处**: https://github.com/huggingface/smolagents/blob/main/docs/source/en/guided_tour.md

原文：
> "Smolagents facilitates the creation of hierarchical multi-agent systems. To achieve this, **agents must have `name` and `description` attributes. These attributes are incorporated into the manager agent's system prompt**, enabling it to understand and invoke the managed agents, similar to how tools are handled."

在 system prompt 模板中的使用方式（来自 [building_good_agents.md](https://github.com/huggingface/smolagents/blob/main/docs/source/en/tutorials/building_good_agents.md)）：

```jinja
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
{%- endfor %}
```

---

## 2. `instructions` - 给 agent 自己看的（追加到该 agent 的 system prompt）

**出处**: https://github.com/huggingface/smolagents/blob/main/docs/source/en/tutorials/building_good_agents.md

原文：
> "It is generally simpler to pass the `instructions` argument upon agent initialization, like `agent = CodeAgent(tools=[], model=InferenceClientModel(model_id=model_id), instructions="Always talk like a 5 year old.")`. **Note that these instructions are appended to the system prompt and do not replace it.**"

还有：
> "Instructions intended for the agent generally can be passed as a string to the `instructions` argument during agent initialization."

---

## 总结

| 参数 | 作用对象 | 说明 |
|------|---------|------|
| `description` | Manager agent 的 system prompt | 让 manager 知道这个 managed agent 能做什么 |
| `instructions` | 该 agent 自己的 system prompt | 追加到该 agent 的 system prompt 末尾，指导其行为 |

