"""
测试 compressed_image_content 函数的功能
Test the compressed_image_content function
"""

import os
import base64
import io
from pathlib import Path
from PIL import Image

# Import the functions to test
from utils.markdown_utils import MarkdownMessage, compressed_image_content


def create_test_image(width: int, height: int, color: tuple = (255, 0, 0)) -> str:
    """创建一个测试图片并返回base64编码的data URL
    Create a test image and return base64-encoded data URL
    """
    img = Image.new('RGB', (width, height), color=color)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def test_compress_large_image():
    """测试压缩大图片
    Test compressing a large image
    """
    print("\n=== Test 1: Compress Large Image ===")

    # 创建一个2000x3000的大图片
    # Create a large 2000x3000 image
    large_image_url = create_test_image(2000, 3000, color=(255, 0, 0))

    # 创建包含大图片的MarkdownMessage
    # Create MarkdownMessage with large image
    original_message = MarkdownMessage([
        {"type": "text", "text": "这是一张大图片:"},
        {"type": "image_url", "image_url": {"url": large_image_url}},
        {"type": "text", "text": "图片说明完毕"}
    ], filename="test_large.md")

    # 压缩图片 (max_short_side_pixels=1080)
    # Compress image (max_short_side_pixels=1080)
    compressed_message = compressed_image_content(original_message, max_short_side_pixels=1080)

    # 验证结果
    # Verify results
    print(f"原始消息: {original_message}")
    print(f"压缩后消息: {compressed_message}")

    # 检查压缩后的图片尺寸
    # Check compressed image size
    for item in compressed_message.content:
        if item.get("type") == "image_url":
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"压缩后图片尺寸: {img.size}")
            print(f"短边: {min(img.size)} (应该 <= 1080)")

            # 断言: 短边应该小于等于1080
            assert min(img.size) <= 1080, f"图片未正确压缩! 短边={min(img.size)}"
            print("✓ 测试通过: 大图片已正确压缩")


def test_small_image_no_compression():
    """测试小图片不被压缩
    Test that small images are not compressed
    """
    print("\n=== Test 2: Small Image (No Compression) ===")

    # 创建一个800x600的小图片
    # Create a small 800x600 image
    small_image_url = create_test_image(800, 600, color=(0, 255, 0))

    # 创建MarkdownMessage
    original_message = MarkdownMessage([
        {"type": "text", "text": "这是一张小图片:"},
        {"type": "image_url", "image_url": {"url": small_image_url}},
    ], filename="test_small.md")

    # 尝试压缩 (max_short_side_pixels=1080)
    compressed_message = compressed_image_content(original_message, max_short_side_pixels=1080)

    # 检查图片尺寸是否保持不变
    for item in compressed_message.content:
        if item.get("type") == "image_url":
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"图片尺寸: {img.size}")

            # 断言: 尺寸应该保持800x600
            assert img.size == (800, 600), f"小图片尺寸被改变! {img.size}"
            print("✓ 测试通过: 小图片尺寸保持不变")


def test_multiple_images():
    """测试处理多张图片
    Test processing multiple images
    """
    print("\n=== Test 3: Multiple Images ===")

    # 创建多张不同尺寸的图片
    image1_url = create_test_image(2000, 1500, color=(255, 0, 0))  # 大图
    image2_url = create_test_image(500, 800, color=(0, 255, 0))    # 小图
    image3_url = create_test_image(3000, 2000, color=(0, 0, 255))  # 大图

    original_message = MarkdownMessage([
        {"type": "text", "text": "图片1:"},
        {"type": "image_url", "image_url": {"url": image1_url}},
        {"type": "text", "text": "图片2:"},
        {"type": "image_url", "image_url": {"url": image2_url}},
        {"type": "text", "text": "图片3:"},
        {"type": "image_url", "image_url": {"url": image3_url}},
    ], filename="test_multiple.md")

    # 压缩
    compressed_message = compressed_image_content(original_message, max_short_side_pixels=1080)

    # 验证所有图片
    image_count = 0
    for item in compressed_message.content:
        if item.get("type") == "image_url":
            image_count += 1
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"图片{image_count} 尺寸: {img.size}, 短边: {min(img.size)}")

            # 断言: 所有图片短边应该 <= 1080
            assert min(img.size) <= 1080, f"图片{image_count}未正确压缩!"

    assert image_count == 3, f"图片数量不对! 期望3张, 实际{image_count}张"
    print("✓ 测试通过: 多张图片处理正确")


def test_text_only_content():
    """测试纯文本内容 (无图片)
    Test text-only content (no images)
    """
    print("\n=== Test 4: Text Only Content ===")

    original_message = MarkdownMessage([
        {"type": "text", "text": "这是纯文本内容，没有图片。"},
        {"type": "text", "text": "第二段文本。"}
    ], filename="test_text.md")

    compressed_message = compressed_image_content(original_message, max_short_side_pixels=1080)

    # 验证内容保持不变
    assert len(compressed_message.content) == 2
    assert all(item["type"] == "text" for item in compressed_message.content)
    print("✓ 测试通过: 纯文本内容保持不变")


def test_custom_compression_size():
    """测试自定义压缩尺寸
    Test custom compression size
    """
    print("\n=== Test 5: Custom Compression Size ===")

    large_image_url = create_test_image(2000, 3000, color=(128, 128, 0))

    original_message = MarkdownMessage([
        {"type": "image_url", "image_url": {"url": large_image_url}}
    ], filename="test_custom.md")

    # 使用自定义的max_short_side_pixels=500
    compressed_message = compressed_image_content(original_message, max_short_side_pixels=500)

    for item in compressed_message.content:
        if item.get("type") == "image_url":
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"压缩后图片尺寸: {img.size}, 短边: {min(img.size)}")

            # 断言: 短边应该 <= 500
            assert min(img.size) <= 500, f"自定义压缩失败! 短边={min(img.size)}"
            assert min(img.size) >= 499, f"压缩过度! 短边={min(img.size)}"  # 允许1像素误差
            print("✓ 测试通过: 自定义压缩尺寸正确")


def test_aspect_ratio_preserved():
    """测试压缩时保持宽高比
    Test that aspect ratio is preserved during compression
    """
    print("\n=== Test 6: Aspect Ratio Preservation ===")

    # 创建一个2000x1000的图片 (宽高比 2:1)
    image_url = create_test_image(2000, 1000, color=(255, 128, 0))

    original_message = MarkdownMessage([
        {"type": "image_url", "image_url": {"url": image_url}}
    ], filename="test_ratio.md")

    compressed_message = compressed_image_content(original_message, max_short_side_pixels=500)

    for item in compressed_message.content:
        if item.get("type") == "image_url":
            url = item["image_url"]["url"]
            base64_part = url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_part)
            img = Image.open(io.BytesIO(img_bytes))

            # 计算宽高比
            ratio = img.size[0] / img.size[1]
            original_ratio = 2000 / 1000  # 2.0

            print(f"原始宽高比: {original_ratio:.2f}")
            print(f"压缩后宽高比: {ratio:.2f}")
            print(f"压缩后尺寸: {img.size}")

            # 断言: 宽高比应该保持一致 (允许小误差)
            assert abs(ratio - original_ratio) < 0.01, f"宽高比未保持! {ratio} vs {original_ratio}"
            print("✓ 测试通过: 宽高比保持正确")


def run_all_tests():
    """运行所有测试
    Run all tests
    """
    print("=" * 60)
    print("开始测试 compressed_image_content 函数")
    print("Testing compressed_image_content function")
    print("=" * 60)

    try:
        test_compress_large_image()
        test_small_image_no_compression()
        test_multiple_images()
        test_text_only_content()
        test_custom_compression_size()
        test_aspect_ratio_preserved()

        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("All tests passed! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
