import json
from jsonschema import Draft7Validator, ValidationError

def run_demo():
    # 1. 定义 Schema (规则)
    # 这是一个电商订单的规则：ID必须是字符串，价格不能为负，商品列表不能为空
    order_schema = {
        "type": "object",
        "required": ["orderId", "totalPrice", "items"],
        "properties": {
            "orderId": {
                "type": "string",
                "pattern": "^ORD-\\d{4}$", # 正则：必须是 ORD- 后面跟4个数字
                "description": "订单ID格式例如：ORD-1234"
            },
            "totalPrice": {
                "type": "number",
                "minimum": 0
            },
            "items": {
                "type": "array",
                "minItems": 1, # 数组不能为空
                "items": {
                    "type": "object",
                    "required": ["name", "qty"],
                    "properties": {
                        "name": {"type": "string"},
                        "qty": {"type": "integer", "minimum": 1}
                    }
                }
            }
        }
    }

    # 2. 定义验证函数
    def validate_data(data_name, json_data):
        print(f"--- 正在验证: {data_name} ---")
        
        # 创建 Draft7Validator 实例
        validator = Draft7Validator(order_schema)
        
        # 收集所有错误
        errors = []
        for error in sorted(validator.iter_errors(json_data), key=lambda e: list(e.path)):
            error_path = '.'.join(str(p) for p in error.path) if error.path else '(root)'
            errors.append({
                'path': error_path,
                'message': error.message,
                'schema_path': '.'.join(str(p) for p in error.schema_path)
            })
        
        # 输出结果
        if len(errors) == 0:
            print("✅ 验证通过！数据符合 Schema 规则。")
        else:
            print(f"❌ 验证失败！发现 {len(errors)} 个错误：")
            for idx, err in enumerate(errors, 1):
                print(f"   错误 {idx}:")
                print(f"      路径: {err['path']}")
                print(f"      信息: {err['message']}")
                print(f"      Schema路径: {err['schema_path']}")
        print("\n")

    # 3. 准备测试数据

    # 数据 A: 完全正确
    valid_data = {
        "orderId": "ORD-2023",
        "totalPrice": 99.5,
        "items": [
            {"name": "Python Book", "qty": 1}
        ]
    }

    # 数据 B: 有错误 (价格是负数，且 items 是空的)
    invalid_data = {
        "orderId": "ORD-2023",
        "totalPrice": -10.0, 
        "items": []
    }
    
    # 数据 C: 格式错误 (orderId 不符合正则)
    bad_format_data = {
        "orderId": "X-123", 
        "totalPrice": 10,
        "items": [{"name": "Book", "qty": 1}]
    }

    # 4. 运行
    validate_data("正确的数据", valid_data)
    validate_data("错误的数据 (负价格+空数组)", invalid_data)
    validate_data("错误的数据 (ID格式不对)", bad_format_data)

if __name__ == "__main__":
    run_demo()