# 将 HAEvaluator 对比图纳入 Feedback（Manifest + 按需读取 + 文本兜底）

## 背景 / 现状
- 目前 `run_llm_ha_gamma.py` 每轮反馈（`feedback`）主要包含：
  - hybrid automaton 的 JSON（HA spec）
  - metrics 文本（`tc/max_diff/mean_diff/...`）
- 但 `utils/Dainarx_code/HA_evaluation.py:HAEvaluator` 实际会生成并保存对比图（真实轨迹 vs simulate 轨迹），该图对后续迭代非常关键：
  - 能直观看到幅值偏差、相位滞后、漂移、局部段误差放大等模式
  - 这些信息往往无法仅靠标量 metrics 推断

## 总目标
在不显著增加 prompt 体积的前提下，把 HAEvaluator 生成的图作为“可引用产物（artifact）”纳入 `feedback`，并让 agent **按需**调用 `hybrid_automaton_image_analysis` 读取这些图片进行分析；同时提供**文本兜底摘要**，保证即便图片不可用，agent 也能继续迭代。

## 设计原则
1. **轻量反馈**：不在 feedback 中内联完整 base64 大图（避免上下文膨胀/截断/成本上升）。
2. **按需取用**：只在 agent 真的需要看曲线差异时才读取图片。
3. **稳定引用**：图片路径与命名应对“多轮迭代/多次运行”稳定、可追溯。
4. **可回退**：即使图片路径失效/无法读取，仍有 `plot_summary` 文本帮助决策。
5. **向后兼容**：不破坏现有 `<image_N>` trace 图分析能力；新能力作为扩展。

## 方案概述（推荐）
### A. Feedback 中新增 “artifact manifest”
在现有 feedback 文本基础上，追加一个结构化 JSON（manifest）区块，包含：
- 本轮迭代号、run_id、metrics（可重复一份，便于机器读取）
- `plot_summary`：对图中差异的文本摘要（兜底）
- `artifacts[]`：图片产物清单（每个 artifact 记录 `id/kind/path/caption/...`）

### B. `hybrid_automaton_image_analysis` 工具扩展支持“文件路径”
当前工具仅支持 `<image_N>`（来自 trace markdown 的内嵌图片）。扩展后支持：
- `image_ref="<image_N>"`：保持现状
- `image_ref="evaluation_results/.../overlay.png"`：新增，从磁盘读取图片 bytes 再送入视觉模型
- （可选）`image_ref="eval_overlay_iter3"`：通过 artifact id 查映射到 path（需要主 agent 注入映射表）

## Feedback Manifest 结构（v1）
> 注：不要求把整个 feedback 改成 JSON；只需要在现有 markdown feedback 末尾附加一个 JSON code block，便于 agent/工具解析。

建议追加一个独立小节，例如：

```markdown
## Evaluation Artifacts (JSON)
```json
{ ... }
```
```

### Top-level 字段建议
```json
{
  "feedback_version": 1,
  "run_id": "20260120_153012_abcd",
  "iteration": 3,
  "metrics": {
    "tc": 0.123,
    "max_diff": 0.456,
    "mean_diff": 0.078,
    "clustering_error": 0
  },
  "plot_summary": "Fallback summary when images are unavailable...",
  "artifacts": [
    {
      "id": "eval_overlay_iter3",
      "kind": "trajectory_overlay",
      "plot_mode": "overlay",
      "path": "evaluation_results/runs/20260120_153012_abcd/iter_3/overlay.png",
      "mime_type": "image/png",
      "caption": "Overlay: ground truth (solid) vs simulated (dash-dot).",
      "created_at": "2026-01-20T15:31:40",
      "sha256": "optional"
    }
  ]
}
```

### `artifacts[].kind` 建议枚举
- `trajectory_overlay`：真实 vs simulate 叠加图（最关键，默认至少包含这一张）
- `trajectory_residual`（可选）：误差曲线/残差图（gt - sim 或 abs error）
- `metrics_report`（可选）：`metrics.txt` 的路径也可作为 artifact（便于外部检索）

## 产物存储布局（建议）
把原先散落在 `evaluation_results/ha_eval_<timestamp>.png` 的命名改成“run/iter”结构，便于稳定引用：

```
evaluation_results/
  runs/
    <run_id>/
      iter_1/
        overlay.png
        metrics.txt
        artifacts.json        (可选：落盘一份 manifest)
      iter_2/
        overlay.png
        metrics.txt
        artifacts.json
      ...
```

### run_id 生成规则
建议在脚本启动时生成一次并贯穿所有 iteration：
- `run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + short_random_suffix`
- 写入 `hyperparameters` 或作为全局变量/主循环变量传入 evaluator 函数

## 具体代码改动点（实现清单）

### 1) `run_llm_ha_gamma.py:evaluate_ha_specification_with_feedback()`
**目标**：除了现有 HA JSON + metrics_text，还要生成并追加 manifest。

建议改动流程：
1. 计算本轮输出目录：`output_dir = evaluation_results/runs/<run_id>/iter_<iteration>/`
2. 调用 evaluator 保存 overlay 图到固定文件名：`overlay.png`
3. 将 metrics 保存到 `metrics.txt`（现有逻辑已有，调整路径即可）
4. 生成 `plot_summary`（文本兜底）：
   - 最小可行：根据 `metrics_dict` 格式化总结（比如 mean/max/tc）
   - 增强版：利用 `evaluator.metrics` 里已有 `simulated_data` / `ground_truth_data`：
     - 计算逐变量 `abs(sim - gt)` 的 `max`/`mean`，选 top-k 最大误差的变量
     - 可选输出“误差峰值发生的时间点 t=...”
5. 构造 `artifacts`：
   - 至少包含 overlay 图条目（`kind="trajectory_overlay"`）
   - `path` 推荐写成**相对路径**（相对 repo root / 当前工作目录），tool 再做解析
6. 把 manifest 以 JSON code block 的形式 append 到 feedback 末尾。
7. （可选）把 manifest 同步写入 `artifacts.json` 落盘，方便外部脚本/调试。

### 2) `utils/imgTools_ha.py:HybridAutomatonImageTool`
**目标**：让 `hybrid_automaton_image_analysis` 支持 `image_ref` 为“文件路径”。

建议扩展 `_extract_image_bytes(image_ref)`：
1. 如果 `image_ref` 匹配 `<image_N>`：走现有逻辑（从 markdown base64 中取 bytes）
2. 否则按“路径”处理：
   - 支持相对路径：以 `os.getcwd()`（或 repo root）为基准 `abspath`
   - 校验文件存在且是图片扩展名（`.png/.jpg/.jpeg/.webp/.gif`）
   - 读取 bytes 返回
3. 安全建议（强烈推荐）：
   - 默认只允许读取 repo 内、或 `evaluation_results/` 内的文件
   - 拒绝 `..` 路径穿越到仓库外
   - 如确有需要可通过环境变量显式放开白名单目录

（可选增强）支持 artifact id：
- 在主 agent 上注入一个 `feedback_artifacts`（`dict[id] = path`）
- tool 先查 id 映射，找不到再按路径处理

### 3) Prompt/说明更新（`obtain_task_and_images()` 生成的任务文本）
**目标**：让 agent 知道“在哪里能找到 evaluator 图，以及如何按需读取”。

在 prompt 中新增说明（示意）：
- “如果上一轮 feedback 中包含 `Evaluation Artifacts (JSON)`，你可以从 `artifacts[].path` 获取对比图路径，然后调用 `hybrid_automaton_image_analysis` 询问差异/误差模式。”

并给 1 个非常短的调用示例（避免 prompt 变长）：
- `hybrid_automaton_image_analysis(image_ref=".../overlay.png", question="Where do simulated and ground truth diverge most?")`

## 与现有迭代记忆（ResultsAggregator）的关系
当前 `ResultsAggregator.get_top_k_feedback()` 主要复述 top-k 的 HA spec + 简要 metrics，不会包含 `result.feedback` 全文。
- 最低成本做法：让“最新一次反馈（Most Recent Attempt）”携带 artifacts（目前主循环已经追加 latest_feedback），即可满足“下一轮按需看图”。
- 可选增强：让 aggregator 也把“最佳若干次尝试”的 artifacts 一并列出，但要控制体积（只列 path+caption，不要内联图）。

## 测试与验收（建议）
### 功能验收
- 跑一轮 `run_llm_ha_gamma.py`：
  - 确认生成目录：`evaluation_results/runs/<run_id>/iter_1/overlay.png`
  - 确认反馈里出现 `Evaluation Artifacts (JSON)` 且 `path` 指向上述文件
- 手动触发一次工具读取：
  - 让 manager agent 调用 `hybrid_automaton_image_analysis`，`image_ref` 使用该 `path`
  - 工具应能读图并返回视觉模型的分析文本

### 失败场景验收（必须）
- 图片文件不存在：tool 返回清晰错误信息（包含期望路径/可用格式提示）
- 路径越界（`../`）：tool 拒绝并提示“path not allowed”

## 可选增强（后续迭代）
1. **Residual plot**：额外输出一张 `abs(sim-gt)` 的误差曲线图，加入 artifacts。
2. **缩略图 fallback**：只在路径不可用时，才在 feedback 里提供一张低 dpi/短边缩放的 base64 缩略图（严格控制大小）。
3. **自动生成图像问诊问题**：基于 metrics 自动生成 1–2 个引导性问题（例如“最大偏差发生在什么时间段？偏差是相位还是幅值？”），鼓励 agent 在需要时调用工具。
4. **产物清理策略**：新增 CLI flag 控制是否清理旧 runs（避免磁盘无限增长）。

## 风险与权衡
- **上下文膨胀**：manifest 必须保持小；每轮默认 1 张图（overlay）即可。
- **路径可访问性**：若将来 manager/managed agent 迁移到远程沙盒（E2B）执行，路径可能不可用；这时可启用“缩略图 fallback”或把产物上传到沙盒。
- **安全性**：工具支持读路径后，需要加白名单限制，防止 agent 读取非预期文件。

## 交付标准（Done Definition）
- feedback 末尾包含可解析的 `Evaluation Artifacts (JSON)`（含 overlay 图 path）。
- `hybrid_automaton_image_analysis` 可以用 `image_ref=<路径>` 成功读取并分析 evaluator 输出图。
- prompt 清楚告知 agent 如何按需加载图片；无图时 `plot_summary` 能提供基本指引。

