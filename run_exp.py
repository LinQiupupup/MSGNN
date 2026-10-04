import subprocess
import sys
import csv
import re
import time
import statistics  # 用于计算均值和标准差

# ================= 配置区域 =================
seeds = [10, 11, 12, 13, 14]
datasets = ['twitch']
csv_file = 'ablation/ablation_twitch_withoutsmooth.csv'
python_exec = sys.executable


# ===========================================

def parse_accuracy(output_log):
    match = re.search(r"Final_Experiment_Result:\s*([\d\.]+)", output_log)
    if match:
        return float(match.group(1))
    match = re.search(r"Test Accuracy.*?([\d\.]+)", output_log, re.DOTALL)
    if match:
        return float(match.group(1))
    return None  # 改为 None 方便后续过滤


def run_experiments():
    total_runs = len(seeds) * len(datasets)
    current_run = 0

    # 用于存储所有成功运行的准确率，按数据集分类
    # 格式: {'cora': [0.81, 0.82, ...]}
    results_map = {d: [] for d in datasets}

    print(f"开始批量实验，总计 {total_runs} 次运行...")

    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Seed', 'Accuracy', 'Status', 'Time_Cost(s)'])

        for dataset in datasets:
            for seed in seeds:
                current_run += 1
                print(f"[{current_run}/{total_runs}] Running: {dataset}, Seed={seed}...", end=" ", flush=True)

                start_time = time.time()
                cmd = [python_exec, 'main.py', '--dataset', dataset, '--seed', str(seed)]

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True)
                    acc = parse_accuracy(result.stdout)
                    duration = round(time.time() - start_time, 2)

                    if acc is not None:
                        results_map[dataset].append(acc)
                        writer.writerow([dataset, seed, acc, 'Success', duration])
                        print(f"Done. Acc: {acc}")
                    else:
                        writer.writerow([dataset, seed, 'N/A', 'Parse Error', duration])
                        print("Parse Error (Acc not found)")

                except Exception as e:
                    print(f"Failed! Error: {str(e)[:50]}")
                    writer.writerow([dataset, seed, 0.0, 'Error', 0])

            # --- 每个数据集跑完后，写入统计结果 ---
            acc_list = results_map[dataset]
            if len(acc_list) > 0:
                avg_acc = round(statistics.mean(acc_list), 4)
                # 如果只有一个数据点，标准差设为 0
                std_dev = round(statistics.pstdev(acc_list), 4) if len(acc_list) > 1 else 0.0

                writer.writerow([])  # 空行分隔
                writer.writerow([f'--- {dataset} Summary ---', 'Average', avg_acc, 'Std Dev', std_dev])
                writer.writerow([])  # 再加个空行
                print(f"\n>> {dataset} 统计结果: Average={avg_acc}, Std={std_dev}\n")

    print(f"所有实验完毕！结果已保存至: {csv_file}")


if __name__ == "__main__":
    run_experiments()