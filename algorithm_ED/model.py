import numpy as np
from typing import Dict, Any, Optional

def process(config: dict, signal: np.ndarray, labels: Optional[np.ndarray] = None, 
            mode: str = 'predict') -> Dict[str, Any]:
    """
    处理函数 - 实际的算法逻辑
    
    Args:
        config: 配置参数
        signal: 输入信号
        labels: 标签（可选）
        mode: 运行模式，'train', 'predict', 'evaluate'
        
    Returns:
        dict: 处理结果
    """
    # 初始化结果字典
    result = {
        "detection_result": None,
        "metrics": {},
        "log": ""
    }
    
    # 提取配置参数
    window_size = config.get("window_size", 1024)
    threshold_factor = config.get("threshold_factor", 1.5)
    use_squared_energy = config.get("use_squared_energy", True)
    feature_selection = config.get("feature_selection", {
        "use_iq": True,
        "use_amplitude": True,
        "use_phase": True,
        "use_spectrum": True
    })
    
    # 处理信号
    try:
        # 验证输入信号
        if not isinstance(signal, np.ndarray):
            raise TypeError(f"输入信号必须是numpy数组，得到: {type(signal)}")
        
        # 信号预处理 - 支持1D、2D和3D信号
        if signal.ndim == 1:
            # 单通道信号，扩展维度
            signal = np.expand_dims(signal, axis=0)
            result["log"] += f"将1维信号扩展为2维，形状: {signal.shape}\n"
        elif signal.ndim == 3:
            # 3D信号通常表示复数信号 (batch_size, signal_length, 2)
            if signal.shape[2] != 2:
                raise ValueError(f"3D信号的第三维度必须为2（实部和虚部），得到: {signal.shape[2]}")
            
            # 将复数信号转换为模值
            signal = np.sqrt(np.sum(signal ** 2, axis=2))
            result["log"] += f"将3维复数信号转换为2维模值信号，形状: {signal.shape}\n"
        elif signal.ndim > 3:
            raise ValueError(f"输入信号维度不能超过3，得到: {signal.ndim}")
        
        batch_size, signal_length = signal.shape
        
        # 验证窗口大小
        if window_size <= 0:
            raise ValueError(f"窗口大小必须大于0，得到: {window_size}")
        if signal_length < window_size:
            raise ValueError(f"信号长度({signal_length})小于窗口大小({window_size})")
        
        # 计算能量
        num_windows = signal_length // window_size
        energy = np.zeros((batch_size, num_windows))
        result["log"] += f"计算能量: 批次大小={batch_size}, 窗口大小={window_size}, 窗口数量={num_windows}\n"
        
        for i in range(batch_size):
            for j in range(0, num_windows * window_size, window_size):
                window = signal[i, j:j + window_size]
                
                # 根据配置选择特征
                if use_squared_energy:
                    window_energy = np.sum(np.abs(window) ** 2)
                else:
                    window_energy = np.sum(np.abs(window))
                
                energy[i, j // window_size] = window_energy
        
        # 计算阈值
        # 在实际应用中，阈值通常基于噪声功率估计
        # 这里简化为使用能量的统计特性
        if mode == 'train' and labels is not None:
            # 在训练模式下，可以基于标签优化阈值
            # 这里简化处理，使用固定阈值因子
            noise_energy = energy[labels == 0]
            if len(noise_energy) > 0:
                threshold = np.mean(noise_energy) * threshold_factor
                result["log"] += f"使用训练模式阈值: {threshold:.4f} (基于噪声能量)\n"
            else:
                threshold = np.mean(energy) * threshold_factor
                result["log"] += f"使用默认阈值: {threshold:.4f} (所有样本均为主用户)\n"
        else:
            # 在预测模式下，使用默认阈值
            threshold = np.mean(energy) * threshold_factor
            result["log"] += f"使用预测模式阈值: {threshold:.4f}\n"
        
        # 能量检测决策
        detection_result = energy > threshold
        result["log"] += f"检测结果形状: {detection_result.shape}, 主用户占比: {np.mean(detection_result):.4f}\n"
        
        # 更新结果
        result["detection_result"] = detection_result
        result["log"] += f"能量检测完成，使用窗口大小: {window_size}, 阈值因子: {threshold_factor}\n"
        
        # 如果有标签，计算评估指标
        if labels is not None:
            # 验证标签
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"标签必须是numpy数组，得到: {type(labels)}")
            
            # 处理标签形状
            if labels.ndim == 2:
                # 假设标签是每批次一个值
                if labels.shape[1] == 1:
                    labels = labels.flatten()
                else:
                    raise ValueError(f"2D标签的第二维度必须为1，得到: {labels.shape[1]}")
            
            # 确保标签长度与批次大小匹配
            if len(labels) != batch_size:
                raise ValueError(f"标签数量({len(labels)})与批次大小({batch_size})不匹配")
            
            # 计算评估指标
            metrics = calculate_metrics(detection_result, labels)
            result["metrics"] = metrics
            result["log"] += f", 检测准确率: {metrics.get('accuracy', 0):.4f}\n"
        
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        result["log"] += f"能量检测处理失败: {str(e)}\n"
        result["log"] += f"错误详情:\n{error_details}\n"
        # 确保返回完整的结果字典
        if result["detection_result"] is None:
            result["detection_result"] = np.array([])
        return result

def calculate_metrics(detection_result: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        detection_result: 检测结果
        labels: 真实标签
        
    Returns:
        dict: 评估指标字典
    """
    # 验证输入
    if not isinstance(detection_result, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("检测结果和标签必须是numpy数组")
    
    batch_size, num_windows = detection_result.shape
    
    # 展平检测结果（每批次一个决策）
    # 这里使用多数投票：如果一个批次中超过一半的窗口检测到主用户，则认为该批次有主用户
    batch_decisions = np.sum(detection_result, axis=1) > (num_windows / 2)
    
    # 确保标签长度与批次大小匹配
    if len(labels) != batch_size:
        raise ValueError(f"标签数量({len(labels)})与批次大小({batch_size})不匹配")
    
    # 计算TP, TN, FP, FN
    tp = np.sum((batch_decisions == 1) & (labels == 1))
    tn = np.sum((batch_decisions == 0) & (labels == 0))
    fp = np.sum((batch_decisions == 1) & (labels == 0))
    fn = np.sum((batch_decisions == 0) & (labels == 1))
    
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算虚警概率和检测概率
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_alarm_rate": false_alarm_rate,
        "detection_rate": detection_rate
    }
