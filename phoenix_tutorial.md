# Phoenix Agent 监控教程

使用 Arize AI Phoenix 记录和监控 smolagents 的运行，完全免费且本地部署。

## 安装依赖

```bash
pip install arize-phoenix openinference-instrumentation-smolagents
```

## 代码集成

在你的 agent 代码开头添加以下内容：

```python
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

register()
SmolagentsInstrumentor().instrument()
```

本项目已在 `run_pure_python_ha.py` 中集成（使用 try/except 包装，未安装时不影响运行）。

## 启动监控服务器

在单独的终端中运行：

```bash
python -m phoenix.server.main serve
```

服务器默认运行在端口 6006。

## 查看监控面板

### 方式 1：VS Code Remote SSH（推荐）

1. 打开 VS Code 底部的 **PORTS** 面板
2. 点击 **Add Port** 按钮
3. 输入 `6006` 并回车
4. 点击 `localhost:6006` 链接，或在本地浏览器访问 http://localhost:6006

### 方式 2：SSH 端口转发

```bash
# 带端口转发的 SSH 连接
ssh -L 6006:localhost:6006 用户名@服务器地址

# 然后在本地浏览器访问
http://localhost:6006
```

### 方式 3：已有 SSH 会话中添加转发

在 SSH 终端中按 `~C`（先按 `~`，再按 `C`），出现 `ssh>` 提示符后输入：

```
-L 6006:localhost:6006
```

## 完整使用流程

```bash
# 终端 1 - 启动 Phoenix 监控服务器
python -m phoenix.server.main serve

# 终端 2 - 运行你的代码
cd /home/phan635/HybridAutomata/baseline_ha/Physics-Supernova-pq
python run_pure_python_ha.py --input-data-path data_all/non_linear/duffing

# 本地浏览器访问监控面板
http://localhost:6006
```

## 监控功能

Phoenix 提供以下监控能力：

- **LLM 调用追踪**：查看每次 LLM 调用的输入/输出
- **工具调用记录**：记录所有工具调用及其参数
- **执行时间统计**：每一步的耗时分析
- **错误追踪**：完整的错误信息和堆栈追踪
- **多 Agent 协作**：可视化 Agent 之间的调用关系

## 参考资料

- [HuggingFace smolagents 监控文档](https://huggingface.co/docs/smolagents/en/tutorials/inspect_runs)
- [Phoenix GitHub](https://github.com/Arize-ai/phoenix)
