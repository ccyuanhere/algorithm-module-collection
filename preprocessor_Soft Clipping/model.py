import numpy as np
from typing import Dict, Optional,Any

def soft_clip(signal: np.ndarray, threshold: float, alpha: float) -> np.ndarray:
    return threshold * np.tanh(signal * alpha / threshold)

def process(
    config: dict,
    signal: np.ndarray,
    labels: Optional[np.ndarray] = None,
    normalization: str = "min_max",
    noise_reduction: bool = False
) -> Dict[str, Any]:
    """
    处理函数 - 实际的算法逻辑

    Args:
        config: 配置参数
        signal: 输入信号
        labels: 标签（可选）

    Returns:
        dict: 处理结果
    """
    try:
        # 从配置获取参数
        threshold = config.get("threshold", 0.6)
        alpha = config.get("alpha", 2.0)
        adaptive = config.get("adaptive_threshold", True)
        channel_specific = config.get("channel_specific", False)

        # 动态调整阈值 - 优化算法
        if adaptive:
            # 使用更可靠的自适应阈值算法
            abs_signal = np.abs(signal)

            # 计算信号的RMS值
            rms = np.sqrt(np.mean(abs_signal**2, axis=-1, keepdims=True))

            # 基于峰值与RMS的比率计算阈值
            peak_to_rms = np.max(abs_signal, axis=-1, keepdims=True) / (rms + 1e-8)
            adaptive_threshold = np.percentile(peak_to_rms, 70) * 0.5

            # 限制阈值范围
            adaptive_threshold = np.clip(adaptive_threshold, 0.4, 0.8)
            threshold = adaptive_threshold
            log_msg = f"Adaptive threshold set to {threshold:.4f}"
        else:
            log_msg = f"Using fixed threshold {threshold}"

        # 预处理：降噪
        if noise_reduction:
            signal = _reduce_noise(signal)
            log_msg += " with noise reduction"

        # 预处理：归一化
        if normalization == "min_max":
            signal = _min_max_normalize(signal)
            log_msg += " and min-max normalization"
        elif normalization == "z_score":
            signal = _z_score_normalize(signal)
            log_msg += " and z-score normalization"

        # 应用软裁剪
        if channel_specific:
            clipped_signal = np.zeros_like(signal)
            for b in range(signal.shape[0]):
                for c in range(signal.shape[1]):
                    clipped_signal[b, c] = soft_clip(signal[b, c], threshold, alpha)
        else:
            clipped_signal = soft_clip(signal, threshold, alpha)

        # 计算性能指标
        metrics = calculate_metrics(signal, clipped_signal, labels, threshold)

        return {
            "output_signal": clipped_signal,
            "metrics": metrics,
            "log": log_msg,
            "success": True
        }
    except Exception as e:
        return {
            "output_signal": None,
            "metrics": {},
            "log": f"Processing error: {str(e)}",
            "success": False
        }

def calculate_metrics(original, clipped, labels, threshold):
    # 基本信号指标
    max_orig = np.max(np.abs(original))
    max_clipped = np.max(np.abs(clipped))

    # 裁剪效果指标
    clipped_points = np.sum(np.abs(clipped) > 0.95 * threshold)
    over_threshold_points = np.sum(np.abs(original) > threshold)

    # 修正裁剪效率计算：限制在[0,1]范围内
    if over_threshold_points > 0:
        clipping_efficiency = min(1.0, clipped_points / over_threshold_points)
    else:
        clipping_efficiency = 0.0

    metrics = {
        "max_value_reduction": max_orig - max_clipped,
        "signal_distortion": np.mean((original - clipped)**2),
        "peak_clipping_ratio": clipped_points / original.size,
        "dynamic_range_reduction": np.ptp(original) - np.ptp(clipped),
        "signal_to_distortion_ratio": 10 * np.log10(
            np.mean(original**2) / (np.mean((original - clipped)**2) + 1e-8)
        ),
        "clipping_efficiency": clipping_efficiency,  # 使用修正后的值
        "threshold_used": threshold,
        "over_threshold_points": over_threshold_points
    }

    # 如果有标签，计算额外指标
    if labels is not None:
        orig_means = np.mean(original, axis=(1, 2))
        clipped_means = np.mean(clipped, axis=(1, 2))

        metrics["label_correlation_original"] = np.corrcoef(labels.flatten(), orig_means)[0, 1]
        metrics["label_correlation_clipped"] = np.corrcoef(labels.flatten(), clipped_means)[0, 1]
        metrics["feature_preservation"] = np.corrcoef(orig_means, clipped_means)[0, 1]

    return metrics

def _reduce_noise(signal: np.ndarray) -> np.ndarray:
    """使用改进的中值滤波降噪"""
    denoised = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            # 自适应窗口大小
            noise_level = np.std(signal[i, j])
            window_size = 5 if noise_level < 0.2 else 7
            denoised[i, j] = _adaptive_median_filter(signal[i, j], max_window=window_size)
    return denoised

def _adaptive_median_filter(x, max_window=7):
    """自适应中值滤波"""
    result = np.zeros_like(x)
    for i in range(len(x)):
        window_size = 3
        while window_size <= max_window:
            start = max(0, i - window_size//2)
            end = min(len(x), i + window_size//2 + 1)
            window = x[start:end]
            median = np.median(window)

            # 检查当前点是否为脉冲噪声
            if (x[i] - median) > 2 * np.std(window):
                result[i] = median
                break
            else:
                window_size += 2
        else:
            result[i] = x[i]
    return result

def _min_max_normalize(signal: np.ndarray) -> np.ndarray:
    """Min-Max归一化"""
    min_val = np.min(signal, axis=-1, keepdims=True)
    max_val = np.max(signal, axis=-1, keepdims=True)
    range_val = max_val - min_val + 1e-8

    # 避免除零错误
    return 2 * ((signal - min_val) / range_val) - 1

def _z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """Z-Score归一化"""
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True) + 1e-8
    return (signal - mean) / std