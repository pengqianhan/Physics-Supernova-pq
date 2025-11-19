# compressed_image_content 函数文档

## 概述

`compressed_image_content` 是一个用于压缩 `MarkdownMessage` 中图片的实用函数。它可以将消息中的高分辨率图片压缩到指定的最大尺寸，以减少内存占用和网络传输开销，同时保持图片的宽高比。

## 函数签名

```python
def compressed_image_content(
    markdownMessage: MarkdownMessage,
    max_short_side_pixels: int = 1080
) -> MarkdownMessage:
    """返回一个新的 MarkdownMessage，其中图片被压缩到 max_short_side_pixels。

    Args:
        markdownMessage: 包含文本和图片的 MarkdownMessage 对象
        max_short_side_pixels: 图片短边的最大像素数，默认为 1080

    Returns:
        新的 MarkdownMessage 对象，包含压缩后的图片
    """
```

## 核心功能

### 1. 图片压缩逻辑

函数会遍历 `MarkdownMessage` 中的所有内容项，对于每个 `image_url` 类型的项：

- **检测图片尺寸**：计算图片的短边（宽度和高度中的较小值）
- **条件压缩**：如果短边 >= `max_short_side_pixels`，则进行压缩
- **保持宽高比**：压缩时按比例缩放，确保宽高比不变
- **重新编码**：将压缩后的图片重新编码为 base64 格式

### 2. 代码示例

#### 基本用法

```python
from utils.markdown_utils import MarkdownMessage, compressed_image_content

# 创建包含图片的 MarkdownMessage
original_message = MarkdownMessage([
    {"type": "text", "text": "问题描述："},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
    {"type": "text", "text": "请分析上图。"}
], filename="problem.md")

# 压缩图片到短边不超过 1080 像素
compressed_message = compressed_image_content(original_message)
```

#### 自定义压缩尺寸

```python
# 压缩到更小的尺寸（短边不超过 500 像素）
compressed_message = compressed_image_content(
    original_message,
    max_short_side_pixels=500
)
```

#### 实际应用场景：ReviewRequestTool

在 `reviewTools.py` 中的实际应用：

```python
class ReviewRequestTool(Tool):
    def forward(self, my_solution: str, my_note: str) -> str:
        # 获取原始高分辨率问题内容
        markdown_content_high_res = self.worker_agent.markdown_content_high_res_image

        # 压缩图片以减少 API 调用开销
        markdown_content: MarkdownMessage = compressed_image_content(
            markdown_content_high_res
        )

        # 构建评审请求
        review_instruction = f"Please review the following solution:\n\n{my_solution}"

        combined_content = [
            {"type": "text", "text": review_instruction}
        ] + markdown_content.content

        # 发送给评审模型
        messages = [
            ChatMessage(role=MessageRole.USER, content=combined_content)
        ]

        return self.review_model.generate(messages).content
```

## 压缩算法详解

### 1. 短边计算

```python
short_side_pixels = min(img.size)  # 取宽度和高度的最小值
```

### 2. 缩放比例计算

```python
if short_side_pixels >= max_short_side_pixels:
    scale = max_short_side_pixels / short_side_pixels
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
```

**示例计算**：
- 原始图片：2000 × 3000 像素
- 短边：min(2000, 3000) = 2000
- max_short_side_pixels = 1080
- 缩放比例：scale = 1080 / 2000 = 0.54
- 新尺寸：(2000 × 0.54, 3000 × 0.54) = (1080, 1620)

### 3. 图片重采样

```python
img = img.resize(new_size, Image.LANCZOS)
```

使用 `LANCZOS` 算法进行高质量重采样。

## 性能优化

### 为什么需要压缩？

1. **减少 API 调用成本**：许多视觉模型按图片大小收费
2. **降低内存占用**：大图片会占用大量内存
3. **加快传输速度**：压缩后的图片传输更快
4. **保持足够质量**：1080px 短边对大多数分析任务已足够

### 压缩效果示例

| 原始尺寸 | 压缩后尺寸 (max=1080) | 压缩比 |
|----------|----------------------|--------|
| 4000×3000 | 1440×1080 | ~56% |
| 2000×1500 | 1440×1080 | ~28% |
| 800×600 | 800×600 (不变) | 0% |

## 错误处理

函数包含完善的错误处理机制：

```python
try:
    # 解码和压缩图片
    base64_part = url.split(",", 1)[1]
    img_bytes = b64decode(base64_part)
    img = Image.open(io.BytesIO(img_bytes))
    # ... 压缩逻辑
except Exception as e:
    # 出错时返回错误文本
    new_content.append({
        "type": "text",
        "text": f"[Error loading image: {str(e)}]"
    })
```

## 特性总结

| 特性 | 说明 |
|------|------|
| **保持宽高比** | ✅ 压缩时自动保持原始宽高比 |
| **条件压缩** | ✅ 仅压缩超过阈值的图片 |
| **多图片支持** | ✅ 可处理包含多张图片的消息 |
| **文本保留** | ✅ 文本内容完全保留，不受影响 |
| **错误恢复** | ✅ 图片处理失败时返回错误信息 |
| **非破坏性** | ✅ 返回新对象，不修改原始数据 |

## 注意事项

1. **不修改原始对象**：函数返回新的 `MarkdownMessage` 对象，原始对象保持不变
2. **仅处理 base64 图片**：仅处理 `data:image` 格式的 base64 编码图片
3. **格式统一**：压缩后的图片统一保存为 PNG 格式
4. **质量损失**：压缩会导致一定的质量损失，但对大多数分析任务影响不大

## 测试

运行测试文件以验证功能：

```bash
python test_compressed_image_content.py
```

测试覆盖以下场景：
- ✅ 大图片压缩
- ✅ 小图片保持不变
- ✅ 多图片处理
- ✅ 纯文本内容
- ✅ 自定义压缩尺寸
- ✅ 宽高比保持

## 相关函数

- `MarkdownMessage`: 数据类，用于存储包含文本和图片的消息
- `markdown_images_compress`: 提取并压缩图片，返回 PIL Image 列表
- `load_markdown_from_filepath`: 从 Markdown 文件加载内容

## 代码位置

- **函数定义**: [utils/markdown_utils.py:27-65](utils/markdown_utils.py#L27-L65)
- **实际应用**: [utils/reviewTools.py:54](utils/reviewTools.py#L54)
- **测试文件**: [test_compressed_image_content.py](test_compressed_image_content.py)
