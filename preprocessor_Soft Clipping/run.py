import numpy as np
import json
import matplotlib.pyplot as plt
import os
from model import process
from typing import Dict, Any, Optional

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    标准运行接口 - 所有算法的统一入口

    Args:
        input_data (dict): 输入数据字典，根据算法类型包含不同字段
        config_path (str, optional): 配置文件路径

    Returns:
        dict: 统一格式的返回结果
        {
            "result": Any,        # 主结果（必需）
            "metrics": dict,      # 可选评估值（可为空字典）
            "log": str,           # 日志信息
            "success": bool       # 是否成功
        }
    """
    try:
        # 确保assets目录存在
        os.makedirs("assets", exist_ok=True)

        # 加载默认配置或指定配置
        config = {}
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)

        # 从input_data获取信号和可选参数
        signal = input_data.get("signal")
        labels = input_data.get("labels")
        normalization = input_data.get("normalization_method", "min_max")
        noise_reduction = input_data.get("noise_reduction", False)

        # 处理信号
        result = process(
            config=config,
            signal=signal,
            labels=labels,
            normalization=normalization,
            noise_reduction=noise_reduction
        )

        # 可视化结果
        if result["success"]:
            visualize_results(signal, result["output_signal"], labels, result["metrics"])

        return {
            "result": result["output_signal"],
            "metrics": result["metrics"],
            "log": result["log"],
            "success": result["success"]
        }
    except Exception as e:
        return {
            "result": None,
            "metrics": {},
            "log": f"Error: {str(e)}",
            "success": False
        }

def visualize_results(original, clipped, labels, metrics):
    """增强版可视化"""
    plt.figure(figsize=(16, 12))

    # 1. 原始信号
    plt.subplot(3, 1, 1)
    sample_idx = 0  # 显示第一个样本
    plt.plot(original[sample_idx, 0, :200], 'b-', label='Original')
    plt.title(f"Original Signal (Sample {sample_idx+1})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 2. 裁剪前后对比
    plt.subplot(3, 1, 2)
    plt.plot(original[sample_idx, 0, :200], 'b-', label='Original')
    plt.plot(clipped[sample_idx, 0, :200], 'r-', label='Clipped')

    # 标记裁剪阈值
    threshold = metrics.get("threshold_used", 0.6)
    plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.5, label='Threshold')
    plt.axhline(y=-threshold, color='g', linestyle='--', alpha=0.5)

    plt.title("Signal Comparison")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 3. 裁剪点分布
    plt.subplot(3, 1, 3)
    clipped_points = np.abs(clipped[sample_idx, 0]) > 0.95 * threshold
    time_points = np.arange(200)

    plt.plot(original[sample_idx, 0, :200], 'b-', label='Original')
    plt.scatter(
        time_points[clipped_points[:200]],
        original[sample_idx, 0, :200][clipped_points[:200]],
        c='red', s=30, label='Clipped Points'
    )

    plt.title("Clipping Points Detection")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/clipping_visualization.png")
    print("Visualization saved to assets/clipping_visualization.png")

    # 4. 性能指标表格
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')

    # 创建表格数据
    metric_data = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_data.append([key, f"{value:.4f}"])
        else:
            metric_data.append([key, str(value)])

    table = ax.table(
        cellText=metric_data,
        colLabels=["Metric", "Value"],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0', '#f0f0f0']
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # 设置标题
    plt.title("Performance Metrics Summary", fontsize=14, pad=20)

    # 高亮重要指标
    for i, key in enumerate(metrics.keys()):
        if key in ["clipping_efficiency", "peak_clipping_ratio", "signal_to_distortion_ratio"]:
            table[(i+1, 0)].set_facecolor('#fffacd')
            table[(i+1, 1)].set_facecolor('#fffacd')

    plt.savefig("assets/metrics_table.png", bbox_inches='tight')
    print("Metrics table saved to assets/metrics_table.png")

if __name__ == "__main__":
    # 示例用法
    print("Running Enhanced Clipping Preprocessor Demo...")

    # 确保数据目录存在
    os.makedirs("data", exist_ok=True)

    # 如果示例数据不存在，生成它们
    if not os.path.exists("data/example_input.npy"):
        print("Generating example data...")
        from make import generate_example_data
        generate_example_data()

    # 加载示例数据
    input_signal = np.load("data/example_input.npy")
    labels = np.load("data/example_labels.npy")

    print(f"Input signal shape: {input_signal.shape}")
    print(f"Labels shape: {labels.shape}")

    # 运行处理
    result = run({
        "signal": input_signal,
        "labels": labels,
        "normalization_method": "min_max",
        "noise_reduction": True
    })

    if result["success"]:
        # 输出性能指标
        print("\nPerformance Metrics:")
        for k, v in result["metrics"].items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
    else:
        print("Processing failed:", result["log"])