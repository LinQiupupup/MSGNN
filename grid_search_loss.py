import optuna
import subprocess
import re
import sys
import csv
import os
import datetime

# ================= 配置区域 =================

# 结果保存文件
CSV_FILE = "optuna_7params_results.csv"

# 设定你想要跑多少次 (建议 100-200 次即可找到极优解)
# 相比穷举的 2000 次，这里只需要 1/10 的时间
N_TRIALS = 150


def write_log(data_row):
    """实时写入 CSV，防止程序中断数据丢失"""
    file_exists = os.path.exists(CSV_FILE)
    # 使用 utf-8-sig 以便 Excel 打开不乱码
    with open(CSV_FILE, 'a' if file_exists else 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头
            header = list(data_row.keys())
            writer.writerow(header)
        writer.writerow(list(data_row.values()))


def objective(trial):
    # ================= 1. 定义搜索空间 (核心) =================

    # --- A. 0.1 到 0.9 的连续参数 (你的重点) ---
    # suggest_float with step=0.1 会在 0.1, 0.2 ... 0.9 中智能选择
    TAU = trial.suggest_float('TAU', 0.1, 0.9, step=0.1)
    beta = trial.suggest_float('beta', 0.1, 0.9, step=0.1)
    gamma = trial.suggest_float('gamma', 0.1, 0.9, step=0.1)

    # --- B. 结构与动力学参数 ---
    # categorical 表示从列表中选一个
    high_threshold = trial.suggest_categorical('high_threshold', [70, 85])
    low_threshold = trial.suggest_categorical('low_threshold', [10, 50])
    eta = trial.suggest_categorical('eta', [1.0, 5.0])
    tau = trial.suggest_categorical('tau', [0.5, 1.0])

    # ================= 2. 固定参数 =================
    # 依然保持你的最佳实践
    fixed_args = [
        "--dataset", "cora",
        "--epochs", "70",  # 快速验证
        "--n_hidden", "512",
        "--lr", "0.001",
        "--dropout", "0.8",
        "--weight_decay", "5e-3",
        "--seed", "20"
    ]

    # ================= 3. 构建命令 =================
    cmd = [sys.executable, "main.py"] + fixed_args
    # 动态添加 Optuna 选出的参数
    cmd.extend(["--TAU", str(TAU)])
    cmd.extend(["--beta", str(beta)])
    cmd.extend(["--gamma", str(gamma)])
    cmd.extend(["--high_threshold", str(high_threshold)])
    cmd.extend(["--low_threshold", str(low_threshold)])
    cmd.extend(["--eta", str(eta)])
    cmd.extend(["--tau", str(tau)])

    accuracy = 0.0

    try:
        # ================= 4. 运行 main.py =================
        # encoding='utf-8', errors='ignore' 防止 Windows 编码崩溃
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='ignore')
        output = result.stdout + result.stderr

        # ================= 5. 提取准确率 =================
        match = re.search(r"Test Accuracy:\s+(\d+\.\d+)", output)
        if match:
            accuracy = float(match.group(1))
        else:
            accuracy = 0.0  # 运行失败或没找到结果

    except Exception as e:
        print(f"Error: {e}")
        accuracy = 0.0

    # ================= 6. 记录日志 =================
    # 构造要保存的数据字典
    record = {
        'Trial_ID': trial.number,
        'Accuracy': accuracy,
        'TAU': TAU,
        'beta': beta,
        'gamma': gamma,
        'high_threshold': high_threshold,
        'low_threshold': low_threshold,
        'eta': eta,
        'tau': tau,
        'State': 'COMPLETE' if accuracy > 0 else 'FAILED'
    }

    # 打印简报
    print(f"Trial {trial.number}: Acc={accuracy:.4f} | T={TAU} b={beta} g={gamma} H={high_threshold} L={low_threshold}")

    # 写入文件
    write_log(record)

    return accuracy


if __name__ == "__main__":
    print(f"=== 开始 Optuna 极速搜索 (目标: 寻找最佳 7 参数组合) ===")
    print(f"结果将保存在: {CSV_FILE}")
    print(f"计划尝试次数: {N_TRIALS} 次 (约穷举法的 1/15 工作量)")
    print("-" * 50)

    # 创建优化器，方向是 maximize (准确率越高越好)
    # TPE Sampler 是默认的，它就是那个“聪明的算法”
    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\n用户手动停止！已保存现有结果。")

    print("\n" + "=" * 50)
    print("搜索结束！")
    if len(study.trials) > 0:
        print(f"最佳准确率: {study.best_value:.4f}")
        print("最佳参数组合:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
    print("=" * 50)