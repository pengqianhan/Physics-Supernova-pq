"""
测试和讲解：hybrid_automaton_image_analysis 如何通过 <iter_image_1_0> 找到图片

整体流程分为两个阶段：
1. 注册阶段（Registration）：在 run_llm_ha_gamma.py 中，评估完成后注册图片
2. 查询阶段（Lookup）：Agent 调用工具时，通过 placeholder 查找图片

关键点：图片路径在注册时就被读取成 bytes 存储，查询时只是通过索引查找 bytes
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.imgTools_ha import HybridAutomatonImageTool


def demo_registration_and_lookup():
    """演示完整的注册和查找流程"""

    print("=" * 70)
    print("阶段 1: 注册阶段 (Registration Phase)")
    print("=" * 70)
    print()

    # 创建工具实例
    tool = HybridAutomatonImageTool()

    # 模拟 run_llm_ha_gamma.py 中的注册逻辑
    # 参考 run_llm_ha_gamma.py 中的代码：
    #
    #   plot_placeholder = f"<iter_image_{iteration}_{file_idx}>"
    #   unique_idx = iteration * 100 + file_idx
    #   image_tool.register_iteration_image(unique_idx, overlay_path)

    iteration = 1
    file_idx = 0

    # 计算 placeholder 和索引
    plot_placeholder = f"<iter_image_{iteration}_{file_idx}>"  # "<iter_image_1_0>"
    unique_idx = iteration * 100 + file_idx  # 1 * 100 + 0 = 100

    print(f"迭代次数 (iteration): {iteration}")
    print(f"文件索引 (file_idx): {file_idx}")
    print(f"生成的 placeholder: {plot_placeholder}")
    print(f"计算的唯一索引: {unique_idx} = {iteration} * 100 + {file_idx}")
    print()

    # 模拟注册一个图片（这里用假的 bytes 数据）
    # 实际中是真实的图片路径，如 "evaluation_results/.../overlay.png"
    fake_image_bytes = b"FAKE_PNG_IMAGE_DATA_FOR_TESTING"

    print("注册图片到工具内部存储:")
    print(f"  tool._iteration_images[{unique_idx}] = <image_bytes>")
    tool._iteration_images[unique_idx] = fake_image_bytes

    print(f"  当前注册的索引列表: {tool.get_registered_iteration_indices()}")
    print()

    print("=" * 70)
    print("阶段 2: 查询阶段 (Lookup Phase)")
    print("=" * 70)
    print()

    # 模拟 Agent 调用工具
    image_ref = "<iter_image_1_0>"
    print(f"Agent 调用: tool.forward('{image_ref}', 'What does this show?')")
    print()

    # 详细展示解析过程
    print("解析步骤:")
    print(f"  1. 输入: '{image_ref}'")

    # 提取内部内容
    inner = image_ref[len("<iter_image_"):-1]
    print(f"  2. 提取内部字符串: '{inner}'")

    # 检查是否包含下划线
    if '_' in inner:
        parts = inner.split('_')
        print(f"  3. 按 '_' 分割: {parts}")
        idx = int(parts[0]) * 100 + int(parts[1])
        print(f"  4. 计算索引: {parts[0]} * 100 + {parts[1]} = {idx}")
    else:
        idx = int(inner)
        print(f"  3. 直接转换为整数: {idx}")

    print(f"  5. 查找: tool._iteration_images[{idx}]")

    # 实际查找
    result_bytes, error = tool._extract_image_bytes_from_placeholder(image_ref)

    if result_bytes:
        print(f"  6. 找到图片! bytes 长度: {len(result_bytes)}")
    else:
        print(f"  6. 错误: {error}")

    print()
    return tool


def demo_with_real_file():
    """使用真实文件演示注册流程"""

    print("=" * 70)
    print("使用真实文件演示")
    print("=" * 70)
    print()

    tool = HybridAutomatonImageTool()

    # 查找一个真实的图片文件用于测试
    test_dirs = [
        "evaluation_results",
        "data_all"
    ]

    test_image_path = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for f in files:
                    if f.endswith('.png'):
                        test_image_path = os.path.join(root, f)
                        break
                if test_image_path:
                    break
        if test_image_path:
            break

    if not test_image_path:
        print("未找到测试图片文件，跳过真实文件测试")
        return

    print(f"找到测试图片: {test_image_path}")
    print()

    # 模拟注册流程（如 run_llm_ha_gamma.py 中）
    iteration = 2
    file_idx = 1
    unique_idx = iteration * 100 + file_idx  # 201
    placeholder = f"<iter_image_{iteration}_{file_idx}>"

    print(f"注册图片:")
    print(f"  iteration={iteration}, file_idx={file_idx}")
    print(f"  unique_idx = {unique_idx}")
    print(f"  placeholder = '{placeholder}'")
    print()

    # 使用 register_iteration_image 方法注册（会读取文件）
    success, error = tool.register_iteration_image(unique_idx, test_image_path)

    if success:
        print(f"✓ 注册成功!")
        print(f"  图片 bytes 已存储在 tool._iteration_images[{unique_idx}]")
        print(f"  存储的 bytes 长度: {len(tool._iteration_images[unique_idx])}")
    else:
        print(f"✗ 注册失败: {error}")
        return

    print()

    # 测试查找
    print(f"测试查找 '{placeholder}':")
    result_bytes, error = tool._extract_image_bytes_from_placeholder(placeholder)

    if result_bytes:
        print(f"✓ 查找成功! 返回 {len(result_bytes)} bytes")
    else:
        print(f"✗ 查找失败: {error}")


def demo_index_mapping():
    """演示不同 placeholder 到索引的映射"""

    print("=" * 70)
    print("Placeholder 到索引的映射关系")
    print("=" * 70)
    print()

    examples = [
        # (placeholder, expected_index, description)
        ("<iter_image_1_0>", 100, "第1次迭代, 第0个文件"),
        ("<iter_image_1_1>", 101, "第1次迭代, 第1个文件"),
        ("<iter_image_1_2>", 102, "第1次迭代, 第2个文件"),
        ("<iter_image_2_0>", 200, "第2次迭代, 第0个文件"),
        ("<iter_image_2_3>", 203, "第2次迭代, 第3个文件"),
        ("<iter_image_5>", 5, "旧格式（向后兼容）"),
        ("<iter_image_100>", 100, "旧格式，索引100"),
    ]

    tool = HybridAutomatonImageTool()

    # 预先注册所有可能的索引
    for _, idx, _ in examples:
        tool._iteration_images[idx] = f"image_data_{idx}".encode()

    print(f"{'Placeholder':<20} {'计算索引':<10} {'说明':<25} {'结果'}")
    print("-" * 70)

    for placeholder, expected_idx, desc in examples:
        result_bytes, error = tool._extract_image_bytes_from_placeholder(placeholder)
        if result_bytes:
            actual_idx = int(result_bytes.decode().split('_')[-1])
            status = "✓" if actual_idx == expected_idx else "✗"
            print(f"{placeholder:<20} {expected_idx:<10} {desc:<25} {status}")
        else:
            print(f"{placeholder:<20} {expected_idx:<10} {desc:<25} ✗ {error}")


def show_internal_structure():
    """展示工具内部数据结构"""

    print("=" * 70)
    print("工具内部数据结构")
    print("=" * 70)
    print()

    print("""
HybridAutomatonImageTool 内部存储结构:

    self._iteration_images: dict[int, bytes] = {}

    这是一个 字典，键是整数索引，值是图片的二进制数据(bytes)

    注册时 (register_iteration_image):
    ┌─────────────────────────────────────────────────────────────┐
    │  overlay_path = "evaluation_results/.../overlay_1_0.png"   │
    │                          ↓                                  │
    │  unique_idx = iteration * 100 + file_idx = 100             │
    │                          ↓                                  │
    │  image_bytes = read_file(overlay_path)  # 读取文件内容      │
    │                          ↓                                  │
    │  _iteration_images[100] = image_bytes   # 存储 bytes       │
    └─────────────────────────────────────────────────────────────┘

    查询时 (_extract_image_bytes_from_placeholder):
    ┌─────────────────────────────────────────────────────────────┐
    │  image_ref = "<iter_image_1_0>"                            │
    │                          ↓                                  │
    │  inner = "1_0"                                              │
    │                          ↓                                  │
    │  parts = ["1", "0"]                                         │
    │                          ↓                                  │
    │  idx = 1 * 100 + 0 = 100                                   │
    │                          ↓                                  │
    │  return _iteration_images[100]  # 返回之前存储的 bytes      │
    └─────────────────────────────────────────────────────────────┘

    关键点：
    - 图片路径 (path) 不会被存储
    - 注册时就将文件内容读取为 bytes 并存储
    - 查询时只是通过计算出的索引查找 bytes
    - 这样设计的好处是：文件只需读取一次，后续查询很快
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("hybrid_automaton_image_analysis 图片查找机制详解")
    print("=" * 70 + "\n")

    # 1. 展示内部数据结构
    show_internal_structure()
    print()

    # 2. 演示注册和查找流程
    demo_registration_and_lookup()
    print()

    # 3. 演示索引映射
    demo_index_mapping()
    print()

    # 4. 使用真实文件演示
    demo_with_real_file()
