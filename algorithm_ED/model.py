import numpy as np
from typing import Dict, Any, Optional

def process(config: dict, signal: np.ndarray, labels: Optional[np.ndarray] = None, 
            mode: str = 'predict') -> Dict[str, Any]:
    """
    能量检测算法处理函数
    
    Args:
        config: 配置参数
        signal: 输入信号，形状为(batch_size, signal_length, 2) [I, Q]
        labels: 标签（仅评估模式需要）
        mode: 运行模式，'predict' 或 'evaluate'
        
    Returns:
        dict: 处理结果
    """
    # 验证模式
    if mode not in ['predict', 'evaluate']:
        raise ValueError(f"不支持的模式: {mode}，支持的模式: predict, evaluate")
    
    # 初始化结果字典
    result = {
        "detection_result": None,
        "metrics": {},
        "log": ""
    }
    
    try:
        # 提取配置参数
        window_size = config.get("window_size", 1024)
        threshold_factor = config.get("threshold_factor", 2.0)
        noise_estimation_method = config.get("noise_estimation_method", "minimum")  # 'minimum', 'percentile'
        
        # 验证输入信号
        if not isinstance(signal, np.ndarray):
            raise TypeError(f"输入信号必须是numpy数组，得到: {type(signal)}")
        
        if signal.ndim != 3 or signal.shape[2] != 2:
            raise ValueError(f"输入信号形状必须为(batch_size, signal_length, 2)，得到: {signal.shape}")
        
        batch_size, signal_length, _ = signal.shape
        result["log"] += f"输入信号形状: {signal.shape}\n"
        
        # 验证窗口大小
        if window_size <= 0 or window_size > signal_length:
            raise ValueError(f"窗口大小必须在(0, {signal_length}]范围内，得到: {window_size}")
        
        # 将I/Q信号转换为复数形式
        signal_complex = signal[:, :, 0] + 1j * signal[:, :, 1]
        
        # 计算窗口化能量
        num_windows = signal_length // window_size
        energy_matrix = np.zeros((batch_size, num_windows))
        
        for i in range(batch_size):
            for j in range(num_windows):
                start_idx = j * window_size
                end_idx = start_idx + window_size
                window = signal_complex[i, start_idx:end_idx]
                
                # 计算窗口能量：|x[n]|²的和
                window_energy = np.sum(np.abs(window) ** 2)
                energy_matrix[i, j] = window_energy
        
        result["log"] += f"计算窗口化能量: 窗口大小={window_size}, 窗口数量={num_windows}\n"
        
        # 噪声底阈值计算
        threshold = calculate_noise_floor_threshold(energy_matrix, threshold_factor, noise_estimation_method)
        
        result["log"] += f"阈值计算方法: noise_floor, 阈值={threshold:.4f}\n"
        
        # 能量检测判决
        detection_result = energy_matrix > threshold
        
        # 计算样本级检测结果（任一窗口检测到信号即认为样本有信号）
        sample_detection = np.any(detection_result, axis=1).astype(int)
        
        result["detection_result"] = sample_detection
        result["log"] += f"完成能量检测，检测到信号的样本数: {np.sum(sample_detection)}/{batch_size}\n"
        
        # 评估模式：计算性能指标
        if mode == 'evaluate':
            if labels is None:
                raise ValueError("评估模式需要提供标签")
            
            if not isinstance(labels, np.ndarray) or len(labels) != batch_size:
                raise ValueError(f"标签形状必须为({batch_size},)，得到: {labels.shape if isinstance(labels, np.ndarray) else type(labels)}")
            
            # 计算评估指标
            metrics = calculate_metrics(sample_detection, labels)
            result["metrics"] = metrics
            
            result["log"] += (
                f"评估结果: 检测率={metrics['detection_rate']:.3f}, "
                f"虚警率={metrics['false_alarm_rate']:.3f}, "
                f"准确率={metrics['accuracy']:.3f}, "
                f"召回率={metrics['recall']:.3f}\n"
            )
        
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        result["log"] += f"能量检测处理失败: {str(e)}\n"
        result["log"] += f"错误详情:\n{error_details}\n"
        
        # 确保返回有效的检测结果
        if result["detection_result"] is None:
            result["detection_result"] = np.zeros(signal.shape[0], dtype=int)
        
        return result

def calculate_noise_floor_threshold(energy_matrix: np.ndarray, threshold_factor: float, 
                                   noise_estimation_method: str) -> float:
    """
    噪声底阈值计算方法
    
    Args:
        energy_matrix: 能量矩阵 (batch_size, num_windows)
        threshold_factor: 阈值因子
        noise_estimation_method: 噪声估计方法，'minimum' 或 'percentile'
        
    Returns:
        float: 计算得到的阈值
    """
    # 将所有窗口能量展平
    all_energies = energy_matrix.flatten()
    
    # 根据噪声估计方法，获取噪声基准
    if noise_estimation_method == "minimum":
        # 使用能量直方图的左峰作为噪声基准
        hist, bin_edges = np.histogram(all_energies, bins=50)
        peak_idx = np.argmax(hist[:len(hist)//2])  # 只在左半部分找峰
        noise_floor = bin_edges[peak_idx]
        
    elif noise_estimation_method == "percentile":
        # 使用20%分位数作为噪声基准
        noise_floor = np.percentile(all_energies, 20)
        
    else:
        raise ValueError(f"不支持的噪声估计方法: {noise_estimation_method}")
    
    # 计算阈值
    effective_factor = max(threshold_factor, 4.0)  # 最小4倍因子
    threshold = noise_floor * effective_factor
    
    return threshold

def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测结果 (batch_size,)
        labels: 真实标签 (batch_size,)
        
    Returns:
        dict: 评估指标字典
    """
    # 计算混淆矩阵
    tp = np.sum((predictions == 1) & (labels == 1))  # 真正例
    tn = np.sum((predictions == 0) & (labels == 0))  # 真负例
    fp = np.sum((predictions == 1) & (labels == 0))  # 假正例
    fn = np.sum((predictions == 0) & (labels == 1))  # 假负例
    
    # 计算各种指标
    total = tp + tn + fp + fn
    
    # 准确率
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # 召回率 (检测率)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    detection_rate = recall
    
    # 虚警率
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        "accuracy": float(accuracy),
        "detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "recall": float(recall),
        "precision": float(precision),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }