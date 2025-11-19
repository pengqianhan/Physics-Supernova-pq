"""
MarkdownMessage 类功能演示
演示如何同时保存文本和图片信息
"""

import base64
import io
from PIL import Image
from utils.markdown_utils import MarkdownMessage, compressed_image_content, markdown_to_plaintext


def create_test_image_base64(width, height, color=(255, 0, 0), text=""):
    """创建测试图片并返回 base64 编码

    Args:
        width: 图片宽度
        height: 图片高度
        color: RGB 颜色元组
        text: 在图片上显示的文字
    """
    from PIL import ImageDraw, ImageFont

    img = Image.new('RGB', (width, height), color=color)

    if text:
        draw = ImageDraw.Draw(img)
        # 使用默认字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()

        # 在图片中心绘制文字
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(position, text, fill=(255, 255, 255), font=font)

    # 转换为 base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def demo_1_basic_usage():
    """演示 1: 基本用法 - 创建包含文本和图片的消息"""
    print("\n" + "="*70)
    print("演示 1: 基本用法 - 同时保存文本和图片")
    print("="*70)

    # 创建包含文本和图片的消息
    message = MarkdownMessage([
        {"type": "text", "text": "问题：一个物体从高处自由落下。"},
        {"type": "text", "text": "参考下图："},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(400, 300, (100, 150, 200), "示意图")}},
        {"type": "text", "text": "请计算物体落地时的速度。"}
    ], filename="physics_problem.md")

    print(f"\n消息对象: {message}")
    print(f"\n文件名: {message.filename}")
    print(f"内容项数量: {len(message.content)}")

    # 分析内容
    text_items = [item for item in message.content if item["type"] == "text"]
    image_items = [item for item in message.content if item["type"] == "image_url"]

    print(f"\n文本段落数: {len(text_items)}")
    print(f"图片数量: {len(image_items)}")

    print("\n文本内容:")
    for i, item in enumerate(text_items, 1):
        print(f"  {i}. {item['text']}")


def demo_2_multiple_images():
    """演示 2: 多图片场景 - 实验对比"""
    print("\n" + "="*70)
    print("演示 2: 多图片场景 - 实验数据对比")
    print("="*70)

    message = MarkdownMessage([
        {"type": "text", "text": "## 实验报告：温度对反应速率的影响\n"},
        {"type": "text", "text": "### 实验组 A (20°C)"},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(600, 400, (255, 100, 100), "20°C")}},
        {"type": "text", "text": "观察结果：反应进行缓慢，10分钟后完成。\n"},
        {"type": "text", "text": "### 实验组 B (40°C)"},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(600, 400, (255, 200, 100), "40°C")}},
        {"type": "text", "text": "观察结果：反应速度加快，5分钟后完成。\n"},
        {"type": "text", "text": "### 实验组 C (60°C)"},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(600, 400, (255, 100, 255), "60°C")}},
        {"type": "text", "text": "观察结果：反应迅速，2分钟后完成。\n"},
        {"type": "text", "text": "### 结论\n温度升高显著提高反应速率。"}
    ], filename="experiment_comparison.md")

    print(f"\n消息对象: {message}")

    # 统计
    text_count = sum(1 for item in message.content if item["type"] == "text")
    image_count = sum(1 for item in message.content if item["type"] == "image_url")

    print(f"\n总共包含:")
    print(f"  - 文本段落: {text_count} 个")
    print(f"  - 实验图片: {image_count} 张")
    print(f"  - 内容总数: {len(message.content)} 项")


def demo_3_text_and_image_extraction():
    """演示 3: 提取文本和图片"""
    print("\n" + "="*70)
    print("演示 3: 分别提取文本和图片")
    print("="*70)

    message = MarkdownMessage([
        {"type": "text", "text": "分析以下两张图片的差异："},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(500, 400, (255, 0, 0), "图A")}},
        {"type": "text", "text": "和"},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(500, 400, (0, 255, 0), "图B")}},
        {"type": "text", "text": "请指出主要区别。"}
    ], filename="comparison.md")

    # 提取所有文本
    all_texts = []
    all_images = []

    for item in message.content:
        if item["type"] == "text":
            all_texts.append(item["text"])
        elif item["type"] == "image_url":
            all_images.append(item["image_url"]["url"])

    print(f"\n提取的文本: {' '.join(all_texts)}")
    print(f"\n图片数量: {len(all_images)}")

    # 解码并查看图片信息
    for i, img_url in enumerate(all_images, 1):
        base64_part = img_url.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_part)
        img = Image.open(io.BytesIO(img_bytes))
        print(f"  图片 {i}: 尺寸 {img.size}, 模式 {img.mode}")


def demo_4_compression():
    """演示 4: 图片压缩功能"""
    print("\n" + "="*70)
    print("演示 4: 压缩大图片")
    print("="*70)

    # 创建包含大图片的消息
    original_message = MarkdownMessage([
        {"type": "text", "text": "这是一张高分辨率图片："},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(3000, 2000, (50, 100, 200), "高清")}},
        {"type": "text", "text": "图片说明完毕。"}
    ], filename="high_res.md")

    print("\n原始消息:")
    print(f"  {original_message}")

    # 检查原始图片大小
    for item in original_message.content:
        if item["type"] == "image_url":
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            original_size = img.size
            original_bytes = len(base64_part)
            print(f"  原始图片尺寸: {img.size}")
            print(f"  Base64 大小: {original_bytes:,} bytes ({original_bytes/1024/1024:.2f} MB)")

    # 压缩图片
    compressed_message = compressed_image_content(original_message, max_short_side_pixels=1080)

    print("\n压缩后的消息:")
    print(f"  {compressed_message}")

    # 检查压缩后的图片大小
    for item in compressed_message.content:
        if item["type"] == "image_url":
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            compressed_size = img.size
            compressed_bytes = len(base64_part)
            print(f"  压缩后尺寸: {img.size}")
            print(f"  Base64 大小: {compressed_bytes:,} bytes ({compressed_bytes/1024/1024:.2f} MB)")

            # 计算压缩比
            reduction = (1 - compressed_bytes / original_bytes) * 100
            print(f"  压缩比: {reduction:.1f}% 减少")


def demo_5_plaintext_conversion():
    """演示 5: 转换为纯文本"""
    print("\n" + "="*70)
    print("演示 5: 提取纯文本（保留图片占位符）")
    print("="*70)

    message = MarkdownMessage([
        {"type": "text", "text": "问题描述："},
        {"type": "text", "text": " <image_1> "},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(400, 300, (200, 100, 50))}},
        {"type": "text", "text": "如图所示，请计算答案。"}
    ], filename="problem.md")

    print(f"\n原始消息: {message}")

    # 转换为纯文本
    plaintext = markdown_to_plaintext(message)

    print(f"\n提取的纯文本:")
    print(f"  '{plaintext}'")
    print(f"\n注意: 图片占位符 <image_1> 被保留在文本中")


def demo_6_openai_compatible():
    """演示 6: OpenAI 兼容格式"""
    print("\n" + "="*70)
    print("演示 6: 用于 LLM API 调用（OpenAI 兼容格式）")
    print("="*70)

    message = MarkdownMessage([
        {"type": "text", "text": "请分析这张图片中的物理现象："},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(600, 400, (100, 200, 150), "物理实验")}},
        {"type": "text", "text": "请详细说明观察到的现象。"}
    ], filename="analysis_request.md")

    print("\n可以直接用于 API 调用:")
    print(f"  message.content 的类型: {type(message.content)}")
    print(f"  message.content 的长度: {len(message.content)}")

    print("\n内容结构（OpenAI/Anthropic 兼容）:")
    for i, item in enumerate(message.content, 1):
        if item["type"] == "text":
            print(f"  {i}. [文本] {item['text'][:50]}...")
        elif item["type"] == "image_url":
            print(f"  {i}. [图片] data:image/png;base64,... (base64 编码)")

    print("\n示例代码:")
    print("""
    from smolagents.models import ChatMessage, MessageRole

    # 直接使用 message.content
    api_message = ChatMessage(
        role=MessageRole.USER,
        content=message.content  # ← 直接使用！
    )

    # 发送给 LLM
    response = model.generate([api_message])
    """)


def demo_7_content_iteration():
    """演示 7: 遍历和处理内容"""
    print("\n" + "="*70)
    print("演示 7: 遍历和处理混合内容")
    print("="*70)

    message = MarkdownMessage([
        {"type": "text", "text": "第一段文字"},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(300, 200, (255, 0, 0), "R")}},
        {"type": "text", "text": "第二段文字"},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(300, 200, (0, 255, 0), "G")}},
        {"type": "image_url", "image_url": {"url": create_test_image_base64(300, 200, (0, 0, 255), "B")}},
        {"type": "text", "text": "第三段文字"}
    ], filename="mixed_content.md")

    print(f"\n消息: {message}")
    print(f"\n详细内容结构:")

    text_count = 0
    image_count = 0

    for i, item in enumerate(message.content, 1):
        if item["type"] == "text":
            text_count += 1
            print(f"  {i}. [文本 #{text_count}] {item['text']}")
        elif item["type"] == "image_url":
            image_count += 1
            # 解码图片查看尺寸
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"  {i}. [图片 #{image_count}] 尺寸: {img.size}")

    print(f"\n统计: {text_count} 段文本, {image_count} 张图片, 共 {len(message.content)} 项")


def demo_8_paired_text_and_images():
    """演示 8: 文字和图片一一对应"""
    print("\n" + "="*70)
    print("演示 8: 多段文字和多张图片一一对应")
    print("="*70)

    # 创建一一对应的文字和图片
    pairs = [
        ("步骤 1: 准备实验材料\n将烧杯放置在实验台上，准备好温度计和搅拌棒。",
         create_test_image_base64(500, 350, (135, 206, 250), "步骤1")),
        ("步骤 2: 加入溶液\n向烧杯中缓慢倒入 100mL 蒸馏水，注意观察液面高度。",
         create_test_image_base64(500, 350, (100, 149, 237), "步骤2")),
        ("步骤 3: 加热溶液\n将烧杯放在酒精灯上，使用小火均匀加热，持续搅拌。",
         create_test_image_base64(500, 350, (255, 165, 0), "步骤3")),
        ("步骤 4: 记录温度\n每隔 30 秒记录一次温度数据，直到溶液沸腾。",
         create_test_image_base64(500, 350, (255, 99, 71), "步骤4")),
        ("步骤 5: 结束实验\n关闭热源，等待溶液冷却后，清理实验器材。",
         create_test_image_base64(500, 350, (60, 179, 113), "步骤5"))
    ]

    # 构建交替的文字-图片内容列表
    content_list = []
    content_list.append({"type": "text", "text": "## 化学实验操作流程\n"})
    
    for text, image_url in pairs:
        content_list.append({"type": "text", "text": text})
        content_list.append({"type": "image_url", "image_url": {"url": image_url}})
    
    content_list.append({"type": "text", "text": "\n## 实验总结\n通过以上 5 个步骤，完成了溶液加热实验。"})

    message = MarkdownMessage(content_list, filename="experiment_steps.md")

    print(f"\n消息: {message}")
    print(f"\n一一对应关系展示:")

    # 提取文字和图片配对
    pair_index = 0
    current_text = None
    
    for item in message.content:
        if item["type"] == "text" and "步骤" in item["text"]:
            current_text = item["text"].split("\n")[0]  # 提取标题
            pair_index += 1
        elif item["type"] == "image_url" and current_text:
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"\n  配对 {pair_index}:")
            print(f"    文字: {current_text}")
            print(f"    图片: 尺寸 {img.size}, 格式 {img.format}")
            current_text = None

    # 统计信息
    text_items = [item for item in message.content if item["type"] == "text"]
    image_items = [item for item in message.content if item["type"] == "image_url"]
    
    print(f"\n统计:")
    print(f"  总文本段落: {len(text_items)} 段")
    print(f"  总图片数量: {len(image_items)} 张")
    print(f"  配对数量: {len(pairs)} 对")
    print(f"  内容总项: {len(message.content)} 项")
    
    print("\n特点:")
    print("  ✓ 每段描述性文字后紧跟一张对应的图片")
    print("  ✓ 文字和图片保持严格的一一对应关系")
    print("  ✓ 便于生成步骤说明、教程、报告等内容")
    print("  ✓ 结构清晰，易于理解和处理")


def run_all_demos():
    """运行所有演示"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MarkdownMessage 类功能演示" + " "*25 + "║")
    print("║" + " "*10 + "演示如何同时保存文本和图片信息" + " "*23 + "║")
    print("╚" + "="*68 + "╝")

    # demo_1_basic_usage()
    # demo_2_multiple_images()
    # demo_3_text_and_image_extraction()
    # demo_4_compression()
    # demo_5_plaintext_conversion()
    # demo_6_openai_compatible()
    # demo_7_content_iteration()
    demo_8_paired_text_and_images()

    print("\n" + "="*70)
    print("所有演示完成！")
    print("="*70)
    print("\n总结:")
    print("  ✓ MarkdownMessage 可以同时保存文本和图片")
    print("  ✓ 支持任意数量的文本段落和图片")
    print("  ✓ 完全兼容 OpenAI/Anthropic API 格式")
    print("  ✓ 提供压缩、提取等实用功能")
    print("  ✓ 友好的字符串显示格式")
    print("  ✓ 支持文字和图片一一对应的配对模式")
    print()


if __name__ == "__main__":
    run_all_demos()
