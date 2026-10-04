import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =====================
# IEEEtran 双栏：推荐全局样式（论文可读字号）
# =====================
mpl.rcParams.update({
    # 让PDF里文字可复制、可编辑（避免Type3字体）
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # 字号：针对 IEEE 双栏图（缩放后仍清晰）
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,

    # 线条略粗一点（印刷更清晰）
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
})

def plot_ablation(dataset: str, baselines, accuracies, stds, outfile_pdf: str):
    # =====================
    # 关键：尺寸按“单栏宽”设计
    # IEEE 双栏常见单栏宽约 3.4 in 左右
    # 你在LaTeX里每张子图 0.48\linewidth 并排 => 这张图整体仍是单栏宽
    # =====================
    fig, ax = plt.subplots(figsize=(4.85, 2.60))  # 单栏宽图：推荐 3.3~3.6 in

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # x 位置
    x = np.arange(len(baselines))

    # 柱子
    ax.bar(
        x, accuracies,
        color=colors[:len(baselines)],
        width=0.82,
        #edgecolor='black',
        linewidth=1.0,
        alpha=0.95,
        zorder=2
    )

    # 误差线：单独画，caps/线宽最稳
    err = ax.errorbar(
        x, accuracies, yerr=stds,
        fmt='none',
        ecolor='#333333',
        elinewidth=2.0,   # ← 误差线“竖线”粗细（你想更粗就调大）
        #capsize=6,        # ← 顶端横杠长度
        #capthick=1.0,     # ← 顶端横杠粗细（若你的版本不支持，删掉这一行）
        zorder=3
    )

    # 如果你的 matplotlib 太老不支持 capthick，会报错：
    # 那就删掉上面 capthick=... 并用下面两行替代加粗 caps：
    # for cap in err[1]:
    #     cap.set_linewidth(2.2)

    # 坐标轴标签
    ax.set_ylabel('Accuracy (%)', labelpad=3)

    # x 轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(baselines)
    ax.tick_params(axis='x', labelsize=11)

    # y 轴范围（按你的数据）
    ax.set_ylim(50, 85)
    ax.set_yticks(np.arange(50, 86, 5))

    # 网格线（论文更好读）
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)

    # 边框（IEEE图通常保留左下，去掉上右更清爽；你想四边都要也行）
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 让布局更紧凑，适合插入论文
    fig.tight_layout(pad=0.2)

    # 保存：PDF（矢量，适合论文）
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # =====================
    # 数据准备（修正反斜杠：用原始字符串 r''，避免 \P 这类转义问题）
    # =====================
    dataset = 'cora'  # 'lastfm' or 'cora'

    if dataset == 'lastfm':
        baselines = ['MSGCN', r'MSGCN\P', r'MSGCN\M', r'MSGCN\S', 'GCN']
        accuracies = np.array([82.37, 81.77, 70.11, 81.76, 77.88])
        stds       = np.array([0.58, 0.33, 0.20, 0.54, 0.88])
    else:
        baselines = ['MSGCN', r'MSGCN\P', r'MSGCN\M', r'MSGCN\S', 'GCN']
        accuracies = np.array([79.36, 77.24, 58.66, 78.70, 71.52])
        stds       = np.array([2.04, 1.18, 4.71, 2.64, 2.84])

    plot_ablation(
        dataset=dataset,
        baselines=baselines,
        accuracies=accuracies,
        stds=stds,
        outfile_pdf=f"ablation_{dataset}.pdf"
    )

    print(f"Saved: ablation_{dataset}.pdf")
