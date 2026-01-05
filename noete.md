问题背景

- 已有一套 hybrid system identification 代码：运行 [main.py](http://main.py/) 生成 Duffing 数据，data/*.npz 输入算法，
输出 Hybrid Automaton（HA）。
- 现方法依赖 meeting_notes.md 中总结的先验配置（order, need_reset, self_loop, kernel, other_items）
并从 JSON 读取，导致通用性与泛化不足。

现状观察

- 用 LLM agent 处理 Duffing 数据(data_duffing/)：能较好识别方程中的非线性项（如 x^3）。
- 但从 Duffing 的 plot 图像中难以分辨模式切换（图像看不出明显切换），导致基于图像的分模困难。
- 对比 Bouncing Ball 的图像（data_plot/ATVA/ball）：存在明显的模式切换特征（如地面接触、速度
反弹）。

核心需求

- 设计一个完整方案，让 LLM agent 直接从 npz 和/或由 npz 绘制的图像中推理出 HA。
- 去除对手动先验的依赖，自动推断并写入 JSON 的 order/need_reset/self_loop/kernel/other_items 等
配置。
- 方案需同时覆盖“两类图像场景”：
    - 图像有明显模式切换（如 Bouncing Ball）
    - 图像无明显模式切换（如 Duffing）

实现偏好与参考

- 借鉴并参考SR-Scientist 的思路与相关文章（程序合成/工具增强/闭环验证）。
- 倾向采用 Hugging Face 的 smolagent 框架实现 HA LLM agent。

交付物

- 将上述完整方案整理为一个 Markdown 文档，包含方法流程、工具设计、搜索与评分策略、两类图像场景的具
体处理细节，以及与现有仓库 JSON/模拟器的兼容性说明。