import json
import os
import numpy as np
import matplotlib.pyplot as plt
from model import process
from pathlib import Path
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
            "log": str,          # 日志信息
            "success": bool      # 是否成功
        }
    """
    try:
        # 加载默认配置
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)

        # 从输入数据获取信号
        signal = input_data["signal"]

        # 处理信号
        result = process(config, signal)

        # 计算性能指标
        metrics = {}
        if "clean_signal" in input_data:  # 修改为检查clean_signal
            noisy_signal = input_data["signal"]
            clean_signal = input_data["clean_signal"]
            denoised_signal = result["denoised_signal"]

            # 确保所有信号都是二维 (batch_size, length)
            noisy_signal = noisy_signal.squeeze()
            clean_signal = clean_signal.squeeze()
            denoised_signal = denoised_signal.squeeze()

            # 如果是一维数组，转换为二维 (1, length)
            if noisy_signal.ndim == 1:
                noisy_signal = noisy_signal[np.newaxis, :]
            if clean_signal.ndim == 1:
                clean_signal = clean_signal[np.newaxis, :]
            if denoised_signal.ndim == 1:
                denoised_signal = denoised_signal[np.newaxis, :]

            # 计算原始信噪比
            noise = noisy_signal - clean_signal
            noise_power = np.mean(noise ** 2, axis=1)
            signal_power = np.mean(clean_signal ** 2, axis=1)
            original_snr = 10 * np.log10(signal_power / noise_power)

            # 计算去噪后信噪比
            residual_noise = denoised_signal - clean_signal
            denoised_noise_power = np.mean(residual_noise ** 2, axis=1)
            denoised_snr = 10 * np.log10(signal_power / denoised_noise_power)

            # 计算SNR提升
            snr_improvement = denoised_snr - original_snr

            # 计算RMSE
            rmse = np.sqrt(np.mean((clean_signal - denoised_signal) ** 2, axis=1))

            metrics = {
                "original_snr": float(np.mean(original_snr)),
                "denoised_snr": float(np.mean(denoised_snr)),
                "snr_improvement": float(np.mean(snr_improvement)),
                "rmse": float(np.mean(rmse))
            }

        return {
            "result": result,
            "metrics": metrics,
            "log": "Processing completed successfully",
            "success": True
        }

    except Exception as e:
        return {
            "result": None,
            "metrics": {},
            "log": f"Error: {str(e)}",
            "success": False
        }

if __name__ == "__main__":
    # 创建assets目录
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    # 加载示例数据
    data_dir = Path(__file__).parent / "data"
    input_signal = np.load(data_dir / "example_input.npy")
    clean_signal = np.load(data_dir / "example_output.npy")

    # 准备输入数据
    input_data = {
        "signal": input_signal,
        "clean_signal": clean_signal  # 用于计算指标
    }

    # 运行算法
    result = run(input_data)

    if result["success"]:
        # 打印性能指标
        print("="*50)
        print("Wavelet Denoising Performance Metrics:")
        print("="*50)
        for k, v in result["metrics"].items():
            print(f"{k.replace('_', ' ').title():<20}: {v:.4f}")
        print("="*50)

        # 获取去噪后的信号
        denoised_signal = result["result"]["denoised_signal"]

        # 确保信号是二维数组 (batch_size, length)
        input_signal = input_signal.squeeze()
        clean_signal = clean_signal.squeeze()
        denoised_signal = denoised_signal.squeeze()

        # 如果是一维数组，转换为二维 (1, length)
        if input_signal.ndim == 1:
            input_signal = input_signal[np.newaxis, :]
        if clean_signal.ndim == 1:
            clean_signal = clean_signal[np.newaxis, :]
        if denoised_signal.ndim == 1:
            denoised_signal = denoised_signal[np.newaxis, :]

        # 可视化结果 - 只展示第一个样本
        sample_idx = 0
        input_sig = input_signal[sample_idx]
        clean_sig = clean_signal[sample_idx]
        denoised_sig = denoised_signal[sample_idx]

        plt.figure(figsize=(12, 8))

        # 原始信号和噪声信号对比
        plt.subplot(2, 1, 1)
        plt.plot(clean_sig, 'b-', linewidth=1.5, label="Clean Signal")
        plt.plot(input_sig, 'r-', alpha=0.6, label="Noisy Signal")
        plt.title("Original Clean Signal vs Noisy Input")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        # 去噪结果对比
        plt.subplot(2, 1, 2)
        plt.plot(clean_sig, 'b-', linewidth=1.5, label="Clean Signal")
        plt.plot(denoised_sig, 'g-', alpha=0.8, label="Denoised Signal")
        plt.title("Clean Signal vs Denoised Output")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(assets_dir / "denoising_comparison.png")
        print(f"Visualization saved to {assets_dir/'denoising_comparison.png'}")

        # 小波系数可视化 - 只展示第一个样本
        coeffs = result["result"]["wavelet_coeffs"]
        if isinstance(coeffs, list):
            levels = len(coeffs) - 1

            plt.figure(figsize=(12, 8))
            for i in range(levels):
                plt.subplot(levels, 1, i+1)
                plt.plot(coeffs[i], 'b-')
                plt.title(f"Wavelet Coefficients - Level {i+1}")
                plt.grid(True)

            plt.subplot(levels, 1, levels)
            plt.plot(coeffs[-1], 'b-')
            plt.title("Approximation Coefficients")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(assets_dir / "wavelet_coefficients.png")
            print(f"Wavelet coefficients saved to {assets_dir/'wavelet_coefficients.png'}")
        else:
            print("Warning: Wavelet coefficients not in expected format for visualization")
    else:
        print("Processing failed:", result["log"])