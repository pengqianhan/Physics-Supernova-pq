# MarkdownMessage 类详细讲解

## 核心概念

`MarkdownMessage` 是一个**自定义数据类型**，专门设计用于同时保存**文本**和**图片**信息，并且完全兼容 OpenAI/Anthropic 等大语言模型的消息格式。

## 类定义

```python
class MarkdownMessage:
    """Custom data type to hold OpenAI-compatible message content with embedded images."""
    def __init__(self, content: List[Dict[str, Any]], filename: str = ""):
        self.content = content      # 核心数据：包含文本和图片的列表
        self.filename = filename    # 可选：来源文件名
```

### 属性说明

| 属性 | 类型 | 说明 |
|------|------|------|
| `content` | `List[Dict[str, Any]]` | 消息内容列表，可包含文本和图片 |
| `filename` | `str` | 可选的文件名，用于标识消息来源 |

## 核心特性：同时保存文本和图片

### 数据结构设计

`content` 是一个**字典列表**，每个字典代表一个内容项，可以是：

#### 1. **文本项** (Text Item)
```python
{
    "type": "text",
    "text": "这是文本内容"
}
```

#### 2. **图片项** (Image Item)
```python
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgoAAAANS..."
    }
}
```

### 混合内容示例

一个 `MarkdownMessage` 可以同时包含多段文本和多张图片：

```python
message = MarkdownMessage([
    {"type": "text", "text": "问题描述："},
    {"type": "text", "text": "请看下图："},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "图1说明：这是第一张图"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "请分析以上两张图的差异。"}
], filename="problem_1.md")
```

**这个消息同时包含**：
- 📝 4 段文本
- 🖼️ 2 张图片（PNG 和 JPEG 格式）

## 完整代码示例

### 示例 1: 创建包含文本和图片的消息

```python
from utils.markdown_utils import MarkdownMessage
import base64

# 1. 准备图片（base64 编码）
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# 2. 创建混合内容
content = [
    {"type": "text", "text": "## 物理问题\n\n一个质量为 10kg 的物体..."},
    {"type": "text", "text": "参考下图："},
    {"type": "image_url", "image_url": {"url": encode_image("diagram.png")}},
    {"type": "text", "text": "请计算物体的加速度。"},
]

# 3. 创建 MarkdownMessage 对象
message = MarkdownMessage(content, filename="physics_problem_1.md")

print(message)
# 输出: MarkdownMessage 'physics_problem_1.md' | text_preview: '## 物理问题  一个质量为 10kg 的物体... 参考下图： 请计算物体的加速度。' | images: <image_1> (total 1)
```

### 示例 2: 访问消息中的文本和图片

```python
# 分别提取文本和图片
texts = []
images = []

for item in message.content:
    if item.get("type") == "text":
        texts.append(item["text"])
    elif item.get("type") == "image_url":
        images.append(item["image_url"]["url"])

print(f"文本段落数: {len(texts)}")  # 4
print(f"图片数量: {len(images)}")    # 1
print(f"所有文本: {' '.join(texts)}")
```

### 示例 3: 从 Markdown 文件加载（自动处理图片）

```python
from utils.markdown_utils import load_markdown_from_filepath

# 自动解析 Markdown 文件中的图片引用
# 例如: ![示意图](./images/diagram.png)
message = load_markdown_from_filepath("problem.md")

# message.content 现在包含:
# - 文本内容
# - 自动编码的图片（base64）
```

## 为什么使用 OpenAI 兼容格式？

### 1. **直接用于 LLM API 调用**

```python
from smolagents.models import ChatMessage, MessageRole

# MarkdownMessage 可以直接用作 LLM 的输入
markdown_msg = MarkdownMessage([
    {"type": "text", "text": "分析这张图片："},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
])

# 创建 API 消息
api_message = ChatMessage(
    role=MessageRole.USER,
    content=markdown_msg.content  # 直接使用 content
)

# 发送给 LLM
response = model.generate([api_message])
```

### 2. **兼容多个平台**

| 平台 | 支持状态 |
|------|---------|
| OpenAI (GPT-4V, GPT-4o) | ✅ 完全兼容 |
| Anthropic (Claude) | ✅ 完全兼容 |
| Google (Gemini) | ✅ 完全兼容 |
| LiteLLM | ✅ 完全兼容 |

## 实际应用场景

### 场景 1: 物理问题求解（在本项目中）

```python
# 在 run_llm_ha_alpha.py 中
class SmolagentsAgent:
    def __init__(self, problem_md_file: str):
        # 加载包含文本说明和图片的问题
        self.markdown_content_high_res_image = load_markdown_from_filepath(problem_md_file)

        # 现在 self.markdown_content_high_res_image 包含:
        # - 问题文本描述
        # - 物理图示（高分辨率）
```

### 场景 2: 评审工具（ReviewRequestTool）

```python
# 在 utils/reviewTools.py 中
class ReviewRequestTool:
    def forward(self, my_solution: str, my_note: str) -> str:
        # 1. 获取原始问题（包含文本和图片）
        original_problem = self.worker_agent.markdown_content_high_res_image

        # 2. 压缩图片
        compressed_problem = compressed_image_content(original_problem)

        # 3. 构建评审请求（包含解答文本 + 原始问题文本和图片）
        review_content = [
            {"type": "text", "text": f"Worker's solution: {my_solution}"}
        ] + compressed_problem.content

        # 4. 发送给评审模型（模型能同时看到文字和图片）
        response = self.review_model.generate([
            ChatMessage(role=MessageRole.USER, content=review_content)
        ])
```

### 场景 3: 多图片对比分析

```python
# 创建包含多张图片的分析请求
analysis_message = MarkdownMessage([
    {"type": "text", "text": "请对比以下三张实验结果图："},
    {"type": "text", "text": "实验组 A:"},
    {"type": "image_url", "image_url": {"url": encode_image("exp_a.png")}},
    {"type": "text", "text": "实验组 B:"},
    {"type": "image_url", "image_url": {"url": encode_image("exp_b.png")}},
    {"type": "text", "text": "对照组:"},
    {"type": "image_url", "image_url": {"url": encode_image("control.png")}},
    {"type": "text", "text": "请分析三组数据的差异。"}
], filename="comparison.md")

# LLM 能够同时看到所有文本说明和三张图片
```

## `__str__` 方法：友好的显示格式

```python
def __str__(self):
    """Return a manager-friendly summary (small, no base64)."""
    texts = [item["text"] for item in self.content if item.get("type") == "text"]
    images = [item["image_url"].get("url", "") for item in self.content if item.get("type") == "image_url"]

    # 文本预览（前 400 字符）
    preview = " ".join(texts)[:400].replace("\n", " ") + ("…" if sum(len(t) for t in texts) > 400 else "")

    # 图片列表（不显示 base64，太长）
    image_list = [f"<image_{idx+1}>" for idx in range(len(images))]

    return (
        f"MarkdownMessage '{self.filename}' | "
        f"text_preview: '{preview}' | "
        f"images: {', '.join(image_list)} (total {len(images)})"
    )
```

### 显示效果

```python
message = MarkdownMessage([
    {"type": "text", "text": "问题：计算加速度"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw..."}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AA..."}}
], filename="problem.md")

print(message)
# 输出:
# MarkdownMessage 'problem.md' | text_preview: '问题：计算加速度' | images: <image_1>, <image_2> (total 2)
```

**优点**：
- ✅ 不显示冗长的 base64 字符串
- ✅ 快速了解消息包含的内容
- ✅ 方便日志记录和调试

## 数据流程图

```
┌─────────────────────┐
│   Markdown 文件      │
│  problem.md         │
│                     │
│  # 问题              │
│  请看下图：          │
│  ![图](img.png)     │
└──────────┬──────────┘
           │
           ↓ load_markdown_from_filepath()
┌─────────────────────────────────────────┐
│        MarkdownMessage                  │
│  ┌───────────────────────────────────┐  │
│  │ content: [                        │  │
│  │   {"type": "text",                │  │
│  │    "text": "# 问题\n请看下图："},  │  │
│  │   {"type": "text",                │  │
│  │    "text": " <image_1> "},       │  │
│  │   {"type": "image_url",           │  │
│  │    "image_url": {                 │  │
│  │      "url": "data:image/png;..."}}│  │
│  │ ]                                 │  │
│  │ filename: "problem.md"            │  │
│  └───────────────────────────────────┘  │
└──────────────┬──────────────────────────┘
               │
               ↓ compressed_image_content()
┌─────────────────────────────────────────┐
│   MarkdownMessage (压缩版)               │
│   (图片短边 ≤ 1080px)                    │
└──────────────┬──────────────────────────┘
               │
               ↓ 用于 LLM API 调用
┌─────────────────────────────────────────┐
│     ChatMessage                         │
│  role: USER                             │
│  content: message.content               │
└─────────────────────────────────────────┘
```

## 与相关函数的配合

### 1. `compressed_image_content()`
```python
# 压缩消息中的图片
compressed_msg = compressed_image_content(original_msg, max_short_side_pixels=1080)
```

### 2. `markdown_to_plaintext()`
```python
# 提取纯文本（用于某些只需要文本的场景）
plain_text = markdown_to_plaintext(message)
# 返回: "问题：计算加速度 <image_1> <image_2>"
```

### 3. `markdown_images_compress()`
```python
# 提取压缩后的 PIL Image 对象列表
pil_images = markdown_images_compress(message, max_short_side_pixels=500)
# 返回: [<PIL.Image.Image>, <PIL.Image.Image>]
```

## 内存和性能考虑

### Base64 编码的开销

```python
# 一张 1MB 的图片
original_size = 1_000_000  # bytes

# Base64 编码后
base64_size = original_size * 4 / 3  # ≈ 1,333,333 bytes

# 在字符串中
string_size = base64_size * 2  # Python 字符串 (Unicode)
# ≈ 2,666,666 bytes = 2.5 MB
```

**这就是为什么需要 `compressed_image_content()`**：
- 将 4000×3000 图片压缩到 1440×1080
- 大小从 ~36MB 降到 ~5MB
- 节省 ~85% 内存和传输开销

## 完整测试示例

让我创建一个测试文件来演示所有功能：

```python
# test_markdown_message.py
from utils.markdown_utils import MarkdownMessage
import base64
from PIL import Image
import io

# 创建测试图片
def create_test_image_base64(width, height, color=(255, 0, 0)):
    img = Image.new('RGB', (width, height), color=color)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# 测试：同时保存文本和图片
message = MarkdownMessage([
    {"type": "text", "text": "## 实验报告"},
    {"type": "text", "text": "### 实验组 A"},
    {"type": "image_url", "image_url": {"url": create_test_image_base64(800, 600, (255, 0, 0))}},
    {"type": "text", "text": "观察：红色样本显示..."},
    {"type": "text", "text": "### 实验组 B"},
    {"type": "image_url", "image_url": {"url": create_test_image_base64(800, 600, (0, 255, 0))}},
    {"type": "text", "text": "观察：绿色样本显示..."},
], filename="experiment_report.md")

print(message)
# 输出: MarkdownMessage 'experiment_report.md' | text_preview: '## 实验报告 ### 实验组 A 观察：红色样本显示... ### 实验组 B 观察：绿色样本显示...' | images: <image_1>, <image_2> (total 2)

# 统计内容
text_count = sum(1 for item in message.content if item["type"] == "text")
image_count = sum(1 for item in message.content if item["type"] == "image_url")

print(f"文本段落: {text_count}")  # 5
print(f"图片数量: {image_count}")  # 2
```

## 总结

`MarkdownMessage` 是一个强大而优雅的数据结构，它：

| 特性 | 说明 |
|------|------|
| ✅ **多模态** | 同时保存文本和图片 |
| ✅ **灵活性** | 可以任意混合文本和图片 |
| ✅ **兼容性** | OpenAI/Anthropic/Gemini 通用格式 |
| ✅ **可扩展** | 轻松添加更多内容项 |
| ✅ **易调试** | `__str__` 提供友好的摘要 |
| ✅ **类型安全** | 使用 Python 类型提示 |

通过这个设计，您可以优雅地处理包含文本和图片的复杂消息，并直接用于 LLM API 调用！
