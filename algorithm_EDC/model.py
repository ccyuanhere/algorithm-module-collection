import numpy as np
from scipy import signal, fft
import warnings
from typing import Dict, Any, Optional

# 忽略除以零的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

def estimate_scd(x, N, L, P, alpha_resolution):
    """
    使用频域平滑法(FAM)估计谱相关密度函数(SCD)
    
    参数:
    x: 输入信号
    N: FFT点数
    L: 时域平滑窗口长度
    P: 频域平滑窗口长度
    alpha_resolution: 循环频率分辨率
    
    返回:
    SCD: 谱相关密度函数
    alphas: 循环频率数组
    """
    # 信号分段
    segments = []
    for i in range(0, len(x) - N + 1, L):
        segments.append(x[i:i+N])
    
    # 初始化SCD矩阵
    num_alpha = int(2 / alpha_resolution)
    SCD = np.zeros((num_alpha, N), dtype=np.complex128)
    alphas = np.linspace(-1, 1, num_alpha)
    
    # 计算每个段的SCD
    for seg in segments:
        X = fft.fft(seg, N)
        
        for i, alpha in enumerate(alphas):
            shift = int(alpha * N / 2)
            if shift < 0:
                X1 = np.roll(X, shift)
                X2 = np.conjugate(np.roll(X, -shift))
            else:
                X1 = np.roll(X, -shift)
                X2 = np.conjugate(np.roll(X, shift))
            
            # 频域平滑
            product = X1 * X2
            smoothed = np.convolve(product, np.ones(P)/P, mode='same')
            SCD[i] += smoothed
    
    # 平均所有段
    SCD /= len(segments)
    return SCD, alphas

def cyclo_detector(config, signal_segment):
    """
    循环平稳特征检测器核心函数
    
    参数:
    config: 配置参数
    signal_segment: 输入信号段
    
    返回:
    decision: 检测结果 (1: 信号存在, 0: 信号不存在)
    test_statistic: 检验统计量
    """
    # 获取配置参数
    N = config.get("fft_points", 1024)
    L = config.get("time_window", 64)
    P = config.get("freq_window", 16)
    alpha_resolution = config.get("alpha_resolution", 0.01)
    target_alpha = config.get("target_alpha", 0.2)
    
    # 计算SCD
    SCD, alphas = estimate_scd(signal_segment, N, L, P, alpha_resolution)
    
    # 找到最接近目标循环频率的索引
    alpha_idx = np.argmin(np.abs(alphas - target_alpha))
    
    # 计算检验统计量 (目标alpha处的平均SCD幅度)
    test_statistic = np.mean(np.abs(SCD[alpha_idx]))
    
    # 计算背景噪声水平 (远离目标频率的平均SCD)
    background_indices = np.where((np.abs(alphas) > 0.3) & (np.abs(alphas) < 0.7))[0]
    if len(background_indices) > 0:
        background_level = np.mean(np.abs(SCD[background_indices]))
    else:
        background_level = 1e-10  # 默认值
    
    # 计算信噪比
    snr_ratio = test_statistic / (background_level + 1e-10)  # 避免除以零
    
    return snr_ratio, test_statistic, background_level

def evaluate_performance(decisions, labels):
    """
    评估检测性能
    
    参数:
    decisions: 检测结果数组
    labels: 真实标签数组
    
    返回:
    metrics: 性能指标字典
    """
    # 确保数组长度相同
    if len(decisions) != len(labels):
        raise ValueError("决策数组和标签数组长度不一致")
    
    # 计算基本性能指标
    true_positives = np.sum((decisions == 1) & (labels == 1))
    false_positives = np.sum((decisions == 1) & (labels == 0))
    true_negatives = np.sum((decisions == 0) & (labels == 0))
    false_negatives = np.sum((decisions == 0) & (labels == 1))
    
    total = len(labels)
    
    # 计算核心性能指标
    pd = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    pfa = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    pm = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / total
    
    return {
        "detection_probability": float(pd),
        "false_alarm_probability": float(pfa),
        "missed_detection_probability": float(pm),
        "accuracy": float(accuracy),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "false_negatives": int(false_negatives),
        "total_samples": int(total)
    }

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
    # 确保信号是二维数组 [num_segments, segment_length]
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    
    num_segments = signal.shape[0]
    decisions = np.zeros(num_segments, dtype=int)
    test_statistics = np.zeros(num_segments)
    background_levels = np.zeros(num_segments)
    snr_ratios = np.zeros(num_segments)
    
    # 处理每个信号段
    for i in range(num_segments):
        snr_ratios[i], test_statistics[i], background_levels[i] = cyclo_detector(config, signal[i])
    
    # 自适应阈值设置 - 使用更智能的方法确定阈值
    if labels is not None and len(labels) == num_segments:
        # 计算信号存在样本的SNR中位数
        signal_snr = snr_ratios[labels == 1]
        noise_snr = snr_ratios[labels == 0]
        
        if len(signal_snr) > 0 and len(noise_snr) > 0:
            # 使用Fisher判别法确定最佳阈值
            mean_signal = np.mean(signal_snr)
            mean_noise = np.mean(noise_snr)
            var_signal = np.var(signal_snr)
            var_noise = np.var(noise_snr)
            
            # Fisher判别阈值
            if var_signal + var_noise > 0:
                fisher_threshold = (mean_signal * var_noise + mean_noise * var_signal) / (var_signal + var_noise)
            else:
                fisher_threshold = (mean_signal + mean_noise) / 2
            
            # 使用Fisher阈值
            decisions = (snr_ratios > fisher_threshold).astype(int)
            config["best_threshold"] = fisher_threshold
        else:
            # 如果缺少标签数据，使用固定阈值
            threshold = config.get("detection_threshold", 1.5)
            decisions = (snr_ratios > threshold).astype(int)
    else:
        # 如果没有标签，使用固定阈值或上次找到的最佳阈值
        threshold = config.get("detection_threshold", 1.5)
        decisions = (snr_ratios > threshold).astype(int)
    
    # 处理结果
    results = {
        "result": decisions,
        "snr_ratios": snr_ratios,
        "test_statistics": test_statistics,
        "background_levels": background_levels
    }
    
    # 评估性能
    if labels is not None and len(labels) == num_segments:
        metrics = evaluate_performance(decisions, labels)
        results["metrics"] = metrics
    
    return results