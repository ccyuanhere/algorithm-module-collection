import numpy as np
import pywt
from typing import Dict, Optional, Any

def process(config: dict, signal: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    处理函数 - 实际的算法逻辑

    Args:
        config: 配置参数
        signal: 输入信号
        labels: 标签（可选）

    Returns:
        dict: 处理结果
    """
    # 获取配置参数
    wavelet = config.get("wavelet", "db8")
    level = config.get("level", 5)
    threshold_mode = config.get("threshold_mode", "soft")
    threshold_multiplier = config.get("threshold_multiplier", 0.1)

    # 保存原始形状用于返回
    original_shape = signal.shape

    # 确保信号是二维的 (batch_size, signal_length)
    if signal.ndim == 3:
        # 合并通道维度
        signal = signal.reshape(signal.shape[0], -1)

    denoised_signals = []
    all_coeffs = []

    for i in range(signal.shape[0]):
        # 小波分解
        coeffs = pywt.wavedec(signal[i], wavelet, level=level)
        all_coeffs.append(coeffs)

        # 计算阈值
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal[i]))) * threshold_multiplier

        # 应用阈值
        thresholded_coeffs = [coeffs[0]]  # 保留近似系数
        for j in range(1, len(coeffs)):
            thresholded_coeffs.append(pywt.threshold(coeffs[j], threshold, mode=threshold_mode))

        # 小波重构
        denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)

        # 裁剪到原始长度
        if len(denoised_signal) > len(signal[i]):
            denoised_signal = denoised_signal[:len(signal[i])]
        elif len(denoised_signal) < len(signal[i]):
            denoised_signal = np.pad(denoised_signal, (0, len(signal[i]) - len(denoised_signal)))

        denoised_signals.append(denoised_signal)

    # 恢复原始形状
    denoised_signals = np.array(denoised_signals)
    if len(original_shape) == 3:
        denoised_signals = denoised_signals.reshape(original_shape)

    return {
        "denoised_signal": np.array(denoised_signals),
        "wavelet_coeffs": all_coeffs[0] if signal.shape[0] == 1 else all_coeffs,
        "wavelet": wavelet,
        "level": level,
        "threshold": threshold
    }