import re
import ast

a="Best Params: {'high_threshold': 95, 'low_threshold': 30, 'k': 3, 'eta': 0.0006394896973610012, 'beta': 3.4635561120345293, 'gamma': 0.03275138143039173, 'tau': 1.2520137380213734, 'lr': 0.0021248341670822785, 'weight_decay': 5.220864377954764e-05, 'dropout': 0.5864742492807596, 'n_hidden': 64}"
def generate_command(log_str):
    # 1. 修改正则以匹配 "Best Params: {...}"
    match = re.search(r"Best Params: ({.*?})", log_str)
    if not match:
        print("未找到参数字典，请检查 log_line 格式是否正确")
        return

    # 2. 将字符串转换为字典
    param_str = match.group(1)
    try:
        params = ast.literal_eval(param_str)
    except Exception as e:
        print(f"解析字典出错: {e}")
        return

    # 3. 构建基础命令 (保留基础配置)
    cmd_parts = ["python main.py"]
    cmd_parts.append("--dataset cora")
    cmd_parts.append("--ptb_rate 0.2")
    cmd_parts.append("--seed 11")

    # 4. 遍历提取到的参数并拼接到命令中
    for key, value in params.items():
        cmd_parts.append(f"--{key} {value}")

    # 5. 生成最终命令
    full_command = " ".join(cmd_parts)
    print("=== 生成的运行命令 (复制下方内容) ===")
    print(full_command)

if __name__ == "__main__":
    generate_command(a)