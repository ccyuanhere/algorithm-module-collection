import numpy as np
from typing import Dict, Any, Optional

def generate_bpsk_template(template_bits: np.ndarray, samples_per_bit: int = 8):
    """
    生成BPSK调制模板信号
    Args:
        template_bits: 比特序列 (例如: [1, -1, 1, 1, -1, -1, 1, -1])
        samples_per_bit: 每个比特的采样点数
    Returns:
        复数模板信号
    """
    # 将比特序列扩展到采样点
    template_signal = np.repeat(template_bits, samples_per_bit).astype(complex)
    return template_signal

def matched_filter_correlation(signal: np.ndarray, template: np.ndarray):
    """
    执行匹配滤波相关运算
    Args:
        signal: 输入信号
        template: 模板信号
    Returns:
        相关值和最大相关位置
    """
    # 计算归一化相关
    correlation = np.correlate(signal, np.conj(template), mode='full')
    
    # 归一化
    signal_energy = np.sqrt(np.sum(np.abs(signal)**2))
    template_energy = np.sqrt(np.sum(np.abs(template)**2))
    
    if signal_energy > 0 and template_energy > 0:
        correlation = correlation / (signal_energy * template_energy)
    
    # 找到最大相关值和位置
    max_corr_idx = np.argmax(np.abs(correlation))
    max_corr_value = np.abs(correlation[max_corr_idx])
    
    return max_corr_value, max_corr_idx, correlation

def adaptive_threshold_estimation(correlation_values: np.ndarray, false_alarm_rate: float = 0.1):
    """
    自适应阈值估计
    Args:
        correlation_values: 相关值数组
        false_alarm_rate: 期望虚警率
    Returns:
        估计的阈值
    """
    # 排序相关值
    sorted_values = np.sort(correlation_values)
    # 根据虚警率确定阈值
    threshold_idx = int((1 - false_alarm_rate) * len(sorted_values))
    threshold = sorted_values[min(threshold_idx, len(sorted_values) - 1)]
    return threshold

def calculate_performance_metrics(detections: np.ndarray, labels: np.ndarray):
    """
    计算性能指标
    Args:
        detections: 检测结果 (0/1)
        labels: 真实标签 (0/1)
    Returns:
        性能指标字典
    """
    # 计算混淆矩阵
    tp = np.sum((labels == 1) & (detections == 1))  # 真正例
    tn = np.sum((labels == 0) & (detections == 0))  # 真负例
    fp = np.sum((labels == 0) & (detections == 1))  # 假正例
    fn = np.sum((labels == 1) & (detections == 0))  # 假负例
    
    # 计算性能指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 也是检测率
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 检测率和虚警率
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # 检测率 = 召回率
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # 虚警率
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'confusion_matrix': {
            'tp': int(tp), 'tn': int(tn), 
            'fp': int(fp), 'fn': int(fn)
        }
    }

def process(config: dict, signal: np.ndarray, labels: Optional[np.ndarray] = None, mode: str = 'predict') -> Dict[str, Any]:
    """
    匹配滤波检测处理函数
    
    Args:
        config: 配置参数
        signal: 输入信号
        labels: 标签（可选）
        mode: 运行模式 ('predict', 'evaluate', 'train')
        
    Returns:
        dict: 处理结果
    """
    try:
        # 获取配置参数
        threshold = config.get('threshold', 0.3)
        template_type = config.get('template_type', 'bpsk')
        samples_per_bit = config.get('samples_per_bit', 8)
        
        # 预定义模板 - BPSK巴克码序列
        if template_type == 'bpsk':
            template_bits = np.array([1, -1, 1, 1, -1, -1, 1, -1])  # 8位巴克码
        else:
            # 可以扩展其他模板类型
            template_bits = np.array([1, -1, 1, 1, -1, -1, 1, -1])
        
        # 生成模板信号
        template_signal = generate_bpsk_template(template_bits, samples_per_bit)
        
        # 确保信号是二维数组
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        # 处理每个信号
        detection_results = []
        detection_values = []
        detection_positions = []
        all_correlations = []
        
        for i in range(signal.shape[0]):
            current_signal = signal[i]
            
            # 执行匹配滤波
            max_corr, max_pos, full_corr = matched_filter_correlation(current_signal, template_signal)
            
            detection_values.append(max_corr)
            detection_positions.append(max_pos)
            all_correlations.append(max_corr)
        
        # 根据模式处理
        if mode == 'predict':
            # 预测模式：直接使用配置的阈值进行检测
            detections = np.array(detection_values) > threshold
            
            return {
                'detections': detections.astype(int),
                'detection_values': detection_values,
                'detection_positions': detection_positions,
                'correlation_threshold': threshold,
                'template_used': template_type
            }
            
        elif mode == 'evaluate':
            # 评估模式：计算性能指标
            if labels is None:
                raise ValueError("evaluate模式需要提供labels")
            
            detections = np.array(detection_values) > threshold
            
            # 计算性能指标
            metrics = calculate_performance_metrics(detections.astype(int), labels)
            
            return {
                'detections': detections.astype(int),
                'detection_values': detection_values,
                'detection_positions': detection_positions,
                'correlation_threshold': threshold,
                'template_used': template_type,
                'metrics': metrics,
                'confusion_matrix': metrics['confusion_matrix']
            }
            
        elif mode == 'train':
            # 训练模式：参数校准
            if labels is None:
                raise ValueError("train模式需要提供labels")
            
            # 使用标签数据校准最优阈值
            # 尝试不同阈值，找到最佳性能
            thresholds = np.linspace(0.1, 0.9, 20)
            best_threshold = threshold
            best_f1 = 0
            best_metrics = {}
            
            for test_threshold in thresholds:
                test_detections = np.array(detection_values) > test_threshold
                test_metrics = calculate_performance_metrics(test_detections.astype(int), labels)
                
                # 使用F1分数作为优化目标
                if test_metrics['f1_score'] > best_f1:
                    best_f1 = test_metrics['f1_score']
                    best_threshold = test_threshold
                    best_metrics = test_metrics
            
            # 使用最佳阈值进行最终检测
            final_detections = np.array(detection_values) > best_threshold
            
            return {
                'detections': final_detections.astype(int),
                'detection_values': detection_values,
                'detection_positions': detection_positions,
                'correlation_threshold': threshold,  # 原始阈值
                'calibrated_threshold': best_threshold,  # 校准后阈值
                'template_used': template_type,
                'calibration_metrics': best_metrics,
                'optimal_template': template_bits.tolist()
            }
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
            
    except Exception as e:
        return {
            'error': str(e),
            'detections': [],
            'detection_values': [],
            'detection_positions': [],
            'correlation_threshold': 0,
            'metrics': {}
        }
