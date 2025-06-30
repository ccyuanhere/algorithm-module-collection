import numpy as np
from scipy import signal as sig
from typing import Dict, Any, Optional
import time
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

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
    # 提取配置参数
    snr = config.get('snr', 3.0)  # 信噪比(dB)
    base_threshold = config.get('base_threshold', 0.5)  # 基础阈值
    threshold_variation = config.get('threshold_variation', 0.2)  # 阈值随机变化范围
    sample_rate = config.get('sample_rate', 1000)  # 采样率(Hz)
    window_size = config.get('window_size', 50)  # 滑动窗口大小
    feature_count = config.get('feature_count', 10)  # 特征数量
    use_machine_learning = config.get('use_machine_learning', False)  # 是否使用机器学习
    model_path = config.get('model_path', 'models/rf_model.pkl')  # 模型保存路径
    
    # 初始化结果字典
    result = {
        "detection_results": [],
        "metrics": {},
        "raw_signal": signal,
        "filtered_signal": None,
        "correlation": None,
        "features": None
    }
    
    # 生成参考信号模板
    template = generate_reference_template(sample_rate)
    
    # 测量处理时间
    start_time = time.time()
    
    # 执行匹配滤波检测
    if len(signal.shape) == 1:
        # 单通道信号处理
        if use_machine_learning:
            # 使用机器学习方法
            detection_result, correlation, filtered, features = ml_match_filter_detection(
                signal, template, model_path, labels, config)
        else:
            # 使用随机阈值方法
            detection_result, correlation, filtered = random_threshold_match_filter_detection(
                signal, template, base_threshold, threshold_variation)
        
        result["detection_results"].append(detection_result)
        result["correlation"] = correlation
        result["filtered_signal"] = filtered
        if use_machine_learning:
            result["features"] = features
    else:
        # 多通道信号处理
        all_features = []
        for i in range(signal.shape[0]):
            if use_machine_learning:
                # 使用机器学习方法
                detection_result, correlation, filtered, features = ml_match_filter_detection(
                    signal[i], template, model_path, labels[i:i+1] if labels is not None else None, config)
                all_features.append(features)
            else:
                # 使用随机阈值方法
                detection_result, correlation, filtered = random_threshold_match_filter_detection(
                    signal[i], template, base_threshold, threshold_variation)
            
            result["detection_results"].append(detection_result)
            if i == 0:
                result["correlation"] = correlation
                result["filtered_signal"] = filtered
        
        if use_machine_learning and len(all_features) > 0:
            result["features"] = np.array(all_features)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 如果提供了标签，计算性能指标
    if labels is not None:
        metrics = calculate_metrics(result["detection_results"], labels, processing_time, snr)
        result["metrics"] = metrics
    
    return result

def generate_reference_template(sample_rate: int, duration: float = 0.1) -> np.ndarray:
    """
    生成参考信号模板
    
    Args:
        sample_rate: 采样率(Hz)
        duration: 模板持续时间(秒)
        
    Returns:
        np.ndarray: 参考信号模板
    """
    t = np.arange(0, duration, 1/sample_rate)
    # 生成一个中心频率为采样率1/10的正弦波作为模板
    freq = sample_rate / 10
    template = np.sin(2 * np.pi * freq * t)
    
    # 添加少量随机性，使模板与原始信号有轻微差异
    template = template * (1 + 0.05 * np.random.randn(len(template)))
    
    # 归一化模板
    template = template / np.sqrt(np.sum(template**2))
    return template

def extract_features(correlation: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    从相关系数中提取特征
    
    Args:
        correlation: 相关系数数组
        window_size: 滑动窗口大小
        
    Returns:
        np.ndarray: 提取的特征
    """
    # 基本统计特征
    max_val = np.max(correlation)
    mean_val = np.mean(correlation)
    std_val = np.std(correlation)
    median_val = np.median(correlation)
    q75_val = np.percentile(correlation, 75)
    q90_val = np.percentile(correlation, 90)
    
    # 峰值特征
    peaks, _ = sig.find_peaks(correlation, height=mean_val)
    peak_count = len(peaks)
    peak_max = np.max(correlation[peaks]) if peak_count > 0 else 0
    peak_mean = np.mean(correlation[peaks]) if peak_count > 0 else 0
    
    # 频谱特征
    fft_vals = np.abs(np.fft.fft(correlation))
    fft_mean = np.mean(fft_vals)
    fft_std = np.std(fft_vals)
    fft_max = np.max(fft_vals)
    
    # 滑动窗口特征
    windowed_features = []
    for i in range(0, len(correlation) - window_size + 1, window_size):
        window = correlation[i:i+window_size]
        windowed_features.extend([
            np.max(window),
            np.mean(window),
            np.std(window),
            np.median(window),
            np.percentile(window, 75),
            np.percentile(window, 90)
        ])
    
    # 组合所有特征
    features = np.array([
        max_val, mean_val, std_val, median_val, q75_val, q90_val,
        peak_count, peak_max, peak_mean, fft_mean, fft_std, fft_max
    ])
    
    # 添加窗口特征的平均值作为最终特征
    if windowed_features:
        windowed_features = np.array(windowed_features)
        windowed_mean = np.mean(windowed_features.reshape(-1, 6), axis=0)
        features = np.concatenate([features, windowed_mean])
    
    return features

def random_threshold_match_filter_detection(signal: np.ndarray, template: np.ndarray, 
                                           base_threshold: float, threshold_variation: float) -> tuple:
    """
    执行带随机阈值的匹配滤波检测
    
    Args:
        signal: 输入信号
        template: 参考信号模板
        base_threshold: 基础阈值
        threshold_variation: 阈值随机变化范围
        
    Returns:
        tuple: (检测结果, 相关系数, 滤波后信号)
    """
    # 执行匹配滤波（使用卷积实现）
    filtered_signal = sig.correlate(signal, template, mode='valid')
    
    # 计算相关系数（归一化）
    signal_energy = np.sqrt(np.sum(signal**2))
    template_energy = np.sqrt(np.sum(template**2))
    correlation = filtered_signal / (signal_energy * template_energy)
    
    # 为每个样本生成随机阈值
    random_factor = np.random.uniform(1 - threshold_variation, 1 + threshold_variation)
    threshold = base_threshold * random_factor
    
    # 应用阈值进行检测
    detection = 1 if np.max(correlation) > threshold else 0
    
    return detection, correlation, filtered_signal

def ml_match_filter_detection(signal: np.ndarray, template: np.ndarray, 
                             model_path: str, labels: Optional[np.ndarray] = None,
                             config: Optional[dict] = None) -> tuple:
    """
    执行基于机器学习的匹配滤波检测
    
    Args:
        signal: 输入信号
        template: 参考信号模板
        model_path: 模型保存路径
        labels: 标签（可选，用于训练）
        config: 配置参数
        
    Returns:
        tuple: (检测结果, 相关系数, 滤波后信号, 特征)
    """
    # 提取配置参数
    window_size = config.get('window_size', 50) if config else 50
    feature_count = config.get('feature_count', 10) if config else 10
    ml_threshold = config.get('ml_threshold', 0.5) if config else 0.5
    
    # 执行匹配滤波
    filtered_signal = sig.correlate(signal, template, mode='valid')
    
    # 计算相关系数
    signal_energy = np.sqrt(np.sum(signal**2))
    template_energy = np.sqrt(np.sum(template**2))
    correlation = filtered_signal / (signal_energy * template_energy)
    
    # 提取特征
    features = extract_features(correlation, window_size)
    
    # 如果有标签，进行训练
    if labels is not None and len(labels) > 0:
        # 确保模型目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 检查模型是否存在
        if os.path.exists(model_path):
            # 加载现有模型
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # 创建新模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 训练模型
        X = features.reshape(1, -1)
        y = labels[0:1]
        model.fit(X, y)
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 使用模型进行预测
        proba = model.predict_proba(X)[0][1]
        detection = 1 if proba > ml_threshold else 0
    else:
        # 检查模型是否存在
        if os.path.exists(model_path):
            # 加载模型进行预测
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            X = features.reshape(1, -1)
            proba = model.predict_proba(X)[0][1]
            detection = 1 if proba > ml_threshold else 0
        else:
            # 如果没有模型，使用简单的阈值方法
            detection = 1 if np.max(correlation) > 0.5 else 0
    
    return detection, correlation, filtered_signal, features

def calculate_metrics(detections: list, labels: np.ndarray, processing_time: float, snr: float) -> Dict[str, float]:
    """
    计算性能指标
    
    Args:
        detections: 检测结果列表
        labels: 真实标签
        processing_time: 处理时间
        snr: 信噪比(dB)
        
    Returns:
        dict: 性能指标字典
    """
    # 确保检测结果和标签长度一致
    min_len = min(len(detections), len(labels))
    detections = detections[:min_len]
    labels = labels[:min_len]
    
    # 计算TP, TN, FP, FN
    tp = np.sum((np.array(detections) == 1) & (np.array(labels) == 1))
    tn = np.sum((np.array(detections) == 0) & (np.array(labels) == 0))
    fp = np.sum((np.array(detections) == 1) & (np.array(labels) == 0))
    fn = np.sum((np.array(detections) == 0) & (np.array(labels) == 1))
    
    # 计算检测概率和虚警概率
    detection_probability = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_probability = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 计算平均检测时间
    avg_detection_time = processing_time / min_len
    
    return {
        "detection_probability": detection_probability,
        "false_alarm_probability": false_alarm_probability,
        "snr": snr,
        "avg_detection_time": avg_detection_time,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }
