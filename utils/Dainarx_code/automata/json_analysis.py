#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计automata目录下所有JSON文件中"input"字段的数量
"""

import json
import os
from pathlib import Path


def count_input_in_json(json_file_path):
    """
    统计单个JSON文件中"input"字段的出现次数

    Args:
        json_file_path: JSON文件路径

    Returns:
        int: "input"字段的数量
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查automaton中是否有input字段
        if 'automaton' in data and 'input' in data['automaton']:
            return 1
        return 0
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return 0


def count_mode_in_json(json_file_path):
    """
    统计单个JSON文件中automaton.mode列表的长度

    Args:
        json_file_path: JSON文件路径

    Returns:
        int: mode列表的长度，如果不存在则返回0
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查automaton中是否有mode字段
        if 'automaton' in data and 'mode' in data['automaton']:
            mode_list = data['automaton']['mode']
            if isinstance(mode_list, list):
                return len(mode_list)
        return 0
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return 0


def analyze_config_in_json(json_file_path):
    """
    分析单个JSON文件中config字段的配置项

    Args:
        json_file_path: JSON文件路径

    Returns:
        dict: 包含配置信息的字典，如果没有config则返回None
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有config字段
        if 'config' in data:
            config = data['config']
            result = {
                'order': config.get('order'),
                'need_reset': config.get('need_reset'),
                'kernel': config.get('kernel'),
                'all_keys': list(config.keys())
            }
            return result
        return None
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None


def analyze_edges_in_json(json_file_path):
    """
    分析单个JSON文件中edge字段的条件表达式

    Args:
        json_file_path: JSON文件路径

    Returns:
        dict: 包含边和条件信息的字典，如果没有edge则返回None
    """
    import re
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有automaton.edge字段
        if 'automaton' in data and 'edge' in data['automaton']:
            edges = data['automaton']['edge']
            
            edge_info = {
                'edge_count': len(edges),
                'edges': [],
                'conditions': [],
                'has_reset': [],
                'operators': set(),
                'variables': set()
            }
            
            # 定义条件中常见的操作符
            operators_pattern = [
                (r'<=', '<='),
                (r'>=', '>='),
                (r'<(?!=)', '<'),
                (r'>(?!=)', '>'),
                (r'==', '=='),
                (r'!=', '!='),
                (r'\band\b', 'and'),
                (r'\bor\b', 'or'),
                (r'\bnot\b', 'not'),
                (r'abs\s*\(', 'abs()'),
            ]
            
            # 变量模式 (如 x, x1, x2, x[0], x[1] 等)
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\[|\s*[<>=!])'
            
            for edge in edges:
                direction = edge.get('direction', 'unknown')
                condition = edge.get('condition', '')
                has_reset = 'reset' in edge
                
                edge_info['edges'].append(direction)
                edge_info['conditions'].append(condition)
                edge_info['has_reset'].append(has_reset)
                
                # 提取操作符
                for pattern, op_name in operators_pattern:
                    if re.search(pattern, condition):
                        edge_info['operators'].add(op_name)
                
                # 提取变量名
                vars_found = re.findall(var_pattern, condition)
                for var in vars_found:
                    # 过滤掉常见的非变量关键词
                    if var not in ['and', 'or', 'not', 'abs', 'True', 'False']:
                        edge_info['variables'].add(var)
            
            # 转换set为list便于JSON序列化
            edge_info['operators'] = list(edge_info['operators'])
            edge_info['variables'] = list(edge_info['variables'])
            
            return edge_info
        return None
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None


def extract_condition_pattern(condition):
    """
    从条件表达式中提取模式类型
    
    Args:
        condition: 条件表达式字符串
        
    Returns:
        list: 识别出的模式类型列表
    """
    patterns = []
    
    # 简单比较: var op value
    if ' <= ' in condition or ' >= ' in condition or ' < ' in condition or ' > ' in condition:
        patterns.append('comparison')
    
    # 等式判断
    if ' == ' in condition or ' != ' in condition:
        patterns.append('equality')
    
    # 复合条件
    if ' and ' in condition:
        patterns.append('compound_and')
    if ' or ' in condition:
        patterns.append('compound_or')
    
    # 函数调用
    if 'abs(' in condition:
        patterns.append('abs_function')
    
    # 变量间比较 (如 x1 - x2 < 3)
    import re
    if re.search(r'[a-zA-Z]\d*\s*-\s*[a-zA-Z]\d*', condition):
        patterns.append('var_difference')
    
    return patterns if patterns else ['simple']


def find_all_json_files(root_dir):
    """
    递归查找所有JSON文件
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        list: 所有JSON文件的路径列表
    """
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def generate_markdown_report(automata_dir, json_files, files_with_input, files_without_input,
                            config_stats, order_stats, need_reset_stats, kernel_stats,
                            all_config_keys, edge_stats=None, mode_stats=None):
    """
    生成Markdown格式的分析报告

    Args:
        automata_dir: automata目录路径
        json_files: 所有JSON文件列表
        files_with_input: 包含input字段的文件列表
        files_without_input: 不包含input字段的文件列表
        config_stats: config配置统计信息
        order_stats: order参数统计信息
        need_reset_stats: need_reset参数统计信息
        kernel_stats: kernel参数统计信息
        all_config_keys: 所有出现过的config键集合
        edge_stats: edge条件统计信息
        mode_stats: mode数量统计信息
    """
    report_path = Path(__file__).parent / "json_analysis_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# JSON Automata Configuration Analysis Report\n\n")
        f.write(f"**扫描目录**: `{automata_dir}`\n\n")
        f.write("---\n\n")

        # 基本统计
        f.write("## 1. 基本统计\n\n")
        f.write(f"- **总JSON文件数**: {len(json_files)}\n")
        f.write(f"- **包含input字段的文件数**: {len(files_with_input)}\n")
        f.write(f"- **不包含input字段的文件数**: {len(files_without_input)}\n")
        f.write(f"- **包含config字段的文件数**: {len(config_stats)}\n\n")

        # Input字段分析
        f.write("## 2. Input字段分析\n\n")
        if files_with_input:
            f.write(f"### 包含input字段的文件 ({len(files_with_input)}):\n\n")
            for file in files_with_input:
                f.write(f"- ✓ `{file}`\n")
            f.write("\n")

        if files_without_input:
            f.write(f"### 不包含input字段的文件 ({len(files_without_input)}):\n\n")
            for file in files_without_input:
                f.write(f"- ✗ `{file}`\n")
            f.write("\n")

        # Order参数统计
        f.write("## 3. Config参数统计\n\n")
        f.write("### 3.1 Order参数\n\n")
        if order_stats:
            f.write("| Order值 | 文件数量 | 文件列表 |\n")
            f.write("|---------|---------|----------|\n")
            for order_val in sorted(order_stats.keys(), key=lambda x: (x is None, x)):
                files = order_stats[order_val]
                files_str = "<br>".join([f"`{f}`" for f in files])
                order_display = "null" if order_val is None else order_val
                f.write(f"| {order_display} | {len(files)} | {files_str} |\n")
        else:
            f.write("*无order参数数据*\n")
        f.write("\n")

        # Need_reset参数统计
        f.write("### 3.2 Need_reset参数\n\n")
        if need_reset_stats:
            f.write("| Need_reset值 | 文件数量 | 文件列表 |\n")
            f.write("|--------------|---------|----------|\n")
            for reset_val in sorted(need_reset_stats.keys(), key=lambda x: (x is None, x)):
                files = need_reset_stats[reset_val]
                files_str = "<br>".join([f"`{f}`" for f in files])
                reset_display = "null" if reset_val is None else str(reset_val)
                f.write(f"| {reset_display} | {len(files)} | {files_str} |\n")
        else:
            f.write("*无need_reset参数数据*\n")
        f.write("\n")

        # Kernel参数统计
        f.write("### 3.3 Kernel参数\n\n")
        if kernel_stats:
            f.write("| Kernel值 | 文件数量 | 文件列表 |\n")
            f.write("|----------|---------|----------|\n")
            for kernel_val in sorted(kernel_stats.keys(), key=lambda x: (x is None, x)):
                files = kernel_stats[kernel_val]
                files_str = "<br>".join([f"`{f}`" for f in files])
                kernel_display = "null" if kernel_val is None else kernel_val
                f.write(f"| {kernel_display} | {len(files)} | {files_str} |\n")
        else:
            f.write("*无kernel参数数据*\n")
        f.write("\n")

        # 所有配置项统计
        f.write("### 3.4 所有配置项汇总\n\n")
        if all_config_keys:
            f.write("在所有JSON文件的config字段中，出现过的所有配置项：\n\n")
            for key in sorted(all_config_keys):
                f.write(f"- `{key}`\n")
        else:
            f.write("*无配置项数据*\n")
        f.write("\n")

        # 详细配置表
        f.write("## 4. 详细配置表\n\n")
        if config_stats:
            f.write("| 文件 | Order | Need_reset | Kernel | 其他配置项 |\n")
            f.write("|------|-------|------------|--------|------------|\n")
            for file_path in sorted(config_stats.keys()):
                config = config_stats[file_path]
                order = "null" if config['order'] is None else config['order']
                need_reset = "null" if config['need_reset'] is None else str(config['need_reset'])
                kernel = "null" if config['kernel'] is None else config['kernel']

                # 其他配置项（排除order, need_reset, kernel）
                other_keys = [k for k in config['all_keys'] if k not in ['order', 'need_reset', 'kernel']]
                other_items = ", ".join([f"`{k}`" for k in other_keys]) if other_keys else "-"

                f.write(f"| `{file_path}` | {order} | {need_reset} | {kernel} | {other_items} |\n")
        else:
            f.write("*无配置数据*\n")
        f.write("\n")

        # Edge条件分析
        f.write("## 5. Edge条件分析\n\n")
        if edge_stats:
            # 基本统计
            total_edges = sum(info['edge_count'] for info in edge_stats.values())
            files_with_edges = len(edge_stats)
            files_with_reset = sum(1 for info in edge_stats.values() if any(info['has_reset']))
            
            f.write("### 5.1 基本统计\n\n")
            f.write(f"- **包含edge字段的文件数**: {files_with_edges}\n")
            f.write(f"- **边的总数**: {total_edges}\n")
            f.write(f"- **包含reset的文件数**: {files_with_reset}\n\n")
            
            # 操作符统计
            f.write("### 5.2 条件操作符统计\n\n")
            all_operators = {}
            for file_path, info in edge_stats.items():
                for op in info['operators']:
                    if op not in all_operators:
                        all_operators[op] = []
                    all_operators[op].append(file_path)
            
            if all_operators:
                f.write("| 操作符 | 出现次数(文件数) | 文件列表 |\n")
                f.write("|--------|------------------|----------|\n")
                for op in sorted(all_operators.keys()):
                    files = all_operators[op]
                    files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                    f.write(f"| `{op}` | {len(files)} | {files_str} |\n")
            f.write("\n")
            
            # 变量统计
            f.write("### 5.3 条件变量统计\n\n")
            all_variables = {}
            for file_path, info in edge_stats.items():
                for var in info['variables']:
                    if var not in all_variables:
                        all_variables[var] = []
                    all_variables[var].append(file_path)
            
            if all_variables:
                f.write("| 变量名 | 出现次数(文件数) | 文件列表 |\n")
                f.write("|--------|------------------|----------|\n")
                for var in sorted(all_variables.keys()):
                    files = all_variables[var]
                    files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                    f.write(f"| `{var}` | {len(files)} | {files_str} |\n")
            f.write("\n")
            
            # 条件模式统计
            f.write("### 5.4 条件模式统计\n\n")
            pattern_stats = {}
            for file_path, info in edge_stats.items():
                for condition in info['conditions']:
                    patterns = extract_condition_pattern(condition)
                    for p in patterns:
                        if p not in pattern_stats:
                            pattern_stats[p] = {'count': 0, 'files': set(), 'examples': []}
                        pattern_stats[p]['count'] += 1
                        pattern_stats[p]['files'].add(file_path)
                        if len(pattern_stats[p]['examples']) < 3:  # 保留最多3个示例
                            pattern_stats[p]['examples'].append(condition)
            
            if pattern_stats:
                f.write("| 模式类型 | 出现次数 | 文件数 | 示例 |\n")
                f.write("|----------|----------|--------|------|\n")
                pattern_descriptions = {
                    'comparison': '比较 (<=, >=, <, >)',
                    'equality': '等式 (==, !=)',
                    'compound_and': '复合条件 (and)',
                    'compound_or': '复合条件 (or)',
                    'abs_function': '绝对值函数 (abs)',
                    'var_difference': '变量差值',
                    'simple': '简单条件'
                }
                for pattern in sorted(pattern_stats.keys()):
                    stats = pattern_stats[pattern]
                    desc = pattern_descriptions.get(pattern, pattern)
                    examples_str = "<br>".join([f"`{e}`" for e in stats['examples']])
                    f.write(f"| {desc} | {stats['count']} | {len(stats['files'])} | {examples_str} |\n")
            f.write("\n")
            
            # 详细边列表
            f.write("### 5.5 详细边列表\n\n")
            f.write("| 文件 | 边数 | 方向 | 条件 | 有Reset |\n")
            f.write("|------|------|------|------|--------|\n")
            for file_path in sorted(edge_stats.keys()):
                info = edge_stats[file_path]
                for i in range(info['edge_count']):
                    direction = info['edges'][i]
                    condition = info['conditions'][i]
                    has_reset = "✓" if info['has_reset'][i] else "✗"
                    # 第一行显示文件名，后续行不显示
                    if i == 0:
                        f.write(f"| `{file_path}` | {info['edge_count']} | {direction} | `{condition}` | {has_reset} |\n")
                    else:
                        f.write(f"| | | {direction} | `{condition}` | {has_reset} |\n")
            f.write("\n")
        else:
            f.write("*无edge数据*\n\n")

        # Mode统计
        f.write("## 6. Mode统计\n\n")
        if mode_stats:
            total_modes = sum(mode_stats.values())
            f.write(f"- **包含mode字段的文件数**: {len(mode_stats)}\n")
            f.write(f"- **Mode的总数**: {total_modes}\n\n")
            
            # 按mode数量分组统计
            mode_count_distribution = {}
            for file_path, count in mode_stats.items():
                if count not in mode_count_distribution:
                    mode_count_distribution[count] = []
                mode_count_distribution[count].append(file_path)
            
            f.write("### 6.1 Mode数量分布\n\n")
            f.write("| Mode数量 | 文件数 | 文件列表 |\n")
            f.write("|---------|--------|----------|\n")
            for count in sorted(mode_count_distribution.keys()):
                files = mode_count_distribution[count]
                files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                f.write(f"| {count} | {len(files)} | {files_str} |\n")
            f.write("\n")
            
            f.write("### 6.2 各文件Mode详情\n\n")
            f.write("| 文件 | Mode数量 |\n")
            f.write("|------|---------|\n")
            for file_path in sorted(mode_stats.keys()):
                f.write(f"| `{file_path}` | {mode_stats[file_path]} |\n")
            f.write("\n")
        else:
            f.write("*无mode数据*\n\n")

    print(f"\n📄 分析报告已保存到: {report_path}")
    return report_path


def main():
    """主函数：统计所有JSON文件中的input字段数量和config配置"""
    # 获取automata目录的路径
    current_dir = Path(__file__).parent
    automata_dir = current_dir

    print(f"扫描目录: {automata_dir}")
    print("=" * 80)

    # 查找所有JSON文件
    json_files = find_all_json_files(automata_dir)
    json_files.sort()  # 排序便于查看

    # 统计每个文件的input字段和config配置
    total_count = 0
    files_with_input = []
    files_without_input = []
    config_stats = {}
    order_stats = {}
    need_reset_stats = {}
    kernel_stats = {}
    all_config_keys = set()
    edge_stats = {}
    mode_stats = {}  # 统计每个文件的mode数量

    for json_file in json_files:
        # 统计input字段
        count = count_input_in_json(json_file)
        total_count += count

        # 获取相对路径便于显示
        rel_path = os.path.relpath(json_file, automata_dir)

        if count > 0:
            files_with_input.append(rel_path)
            print(f"✓ {rel_path}")
        else:
            files_without_input.append(rel_path)
            print(f"✗ {rel_path}")

        # 分析config配置
        config_info = analyze_config_in_json(json_file)
        if config_info:
            config_stats[rel_path] = config_info

            # 统计order值
            order_val = config_info['order']
            if order_val not in order_stats:
                order_stats[order_val] = []
            order_stats[order_val].append(rel_path)

            # 统计need_reset值
            need_reset_val = config_info['need_reset']
            if need_reset_val not in need_reset_stats:
                need_reset_stats[need_reset_val] = []
            need_reset_stats[need_reset_val].append(rel_path)

            # 统计kernel值
            kernel_val = config_info['kernel']
            if kernel_val not in kernel_stats:
                kernel_stats[kernel_val] = []
            kernel_stats[kernel_val].append(rel_path)

            # 收集所有配置键
            all_config_keys.update(config_info['all_keys'])

        # 分析edge条件
        edge_info = analyze_edges_in_json(json_file)
        if edge_info:
            edge_stats[rel_path] = edge_info

        # 统计mode数量
        mode_count = count_mode_in_json(json_file)
        if mode_count > 0:
            mode_stats[rel_path] = mode_count

    # 打印统计结果
    print("=" * 80)
    print(f"\n统计结果:")
    print(f"  总JSON文件数: {len(json_files)}")
    print(f"  包含input字段的文件数: {len(files_with_input)}")
    print(f"  不包含input字段的文件数: {len(files_without_input)}")
    print(f"  包含config字段的文件数: {len(config_stats)}")
    print(f"\nConfig参数统计:")
    print(f"  Order参数分布: {dict((k, len(v)) for k, v in order_stats.items())}")
    print(f"  Need_reset参数分布: {dict((k, len(v)) for k, v in need_reset_stats.items())}")
    print(f"  Kernel参数分布: {dict((k, len(v)) for k, v in kernel_stats.items())}")
    print(f"  所有配置项: {sorted(all_config_keys)}")

    # 打印Edge统计结果
    print(f"\nEdge条件统计:")
    print(f"  包含edge字段的文件数: {len(edge_stats)}")
    total_edges = sum(info['edge_count'] for info in edge_stats.values())
    print(f"  边的总数: {total_edges}")
    
    # 统计所有操作符
    all_operators = set()
    all_variables = set()
    for info in edge_stats.values():
        all_operators.update(info['operators'])
        all_variables.update(info['variables'])
    print(f"  使用的操作符: {sorted(all_operators)}")
    print(f"  使用的变量: {sorted(all_variables)}")

    # 打印Mode统计结果
    print(f"\nMode统计:")
    print(f"  包含mode字段的文件数: {len(mode_stats)}")
    total_modes = sum(mode_stats.values())
    print(f"  Mode的总数: {total_modes}")
    print(f"  各文件Mode数量:")
    for file_path, count in sorted(mode_stats.items()):
        print(f"    {file_path}: {count}")

    # 生成Markdown报告
    generate_markdown_report(automata_dir, json_files, files_with_input, files_without_input,
                            config_stats, order_stats, need_reset_stats, kernel_stats,
                            all_config_keys, edge_stats, mode_stats)


if __name__ == "__main__":
    main()

