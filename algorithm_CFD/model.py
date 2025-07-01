import numpy as np
import os
import pickle
from scipy import signal as sig
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional

def compute_cyclic_spectrum(signal, fs, alpha, nfft=1024):
    """
    计算循环谱密度
    
    Args:
        signal: 输入信号
        fs: 采样频率
        alpha: 循环频率
        nfft: FFT点数
    
    Returns:
        循环谱密度
    """
    # 计算短时傅里叶变换
    f, t, Zxx = sig.stft(signal, fs=fs, nperseg=nfft, noverlap=nfft//2, nfft=nfft)
    
    # 计算循环谱密度
    csd = np.zeros_like(Zxx, dtype=complex)
    for i in range(len(t)):
        for j in range(len(f)):
            # 计算循环频率偏移
            offset = int(alpha * nfft / fs)
            if 0 <= j + offset < len(f):
                csd[j, i] = Zxx[j, i] * np.conj(Zxx[j + offset, i])
    
    return csd

def extract_features(csd, alpha_range):
    """
    从循环谱密度中提取特征
    
    Args:
        csd: 循环谱密度
        alpha_range: 循环频率范围
    
    Returns:
        提取的特征
    """
    # 计算特征 - 这里使用了简单的统计特征
    features = []
    
    # 计算每个循环频率的能量
    for alpha in alpha_range:
        energy = np.sum(np.abs(csd)**2)
        features.append(energy)
    
    # 计算其他统计特征
    features.append(np.mean(np.abs(csd)))
    features.append(np.std(np.abs(csd)))
    features.append(np.max(np.abs(csd)))
    features.append(np.min(np.abs(csd)))
    
    return np.array(features)

def train_model(features, labels, model_path):
    """
    训练随机森林分类器
    
    Args:
        features: 特征
        labels: 标签
        model_path: 模型保存路径
    
    Returns:
        训练好的模型
    """
    # 创建随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 训练模型
    model.fit(features, labels)
    
    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

def load_model(model_path):
    """
    加载预训练模型
    
    Args:
        model_path: 模型路径
    
    Returns:
        加载的模型
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def process(config: dict, signal: np.ndarray, labels: Optional[np.ndarray] = None, mode: str = "predict") -> Dict[str, Any]:
    """
    处理函数 - 实际的算法逻辑

    Args:
        config: 配置参数
        signal: 输入信号
        labels: 标签（可选）
        mode: 运行模式："train", "predict", "evaluate"

    Returns:
        dict: 处理结果
    """
    # 配置参数
    fs = config.get("sample_rate", 1000)
    nfft = config.get("nfft", 1024)
    alpha_range = np.linspace(config.get("alpha_min", 0.1), 
                              config.get("alpha_max", 0.5), 
                              config.get("alpha_steps", 10))
    
    # 模型路径
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'cfd_model.pkl')
    
    # 初始化结果
    result = {
        "result": None,
        "metrics": {},
        "log": ""
    }
    
    # 处理信号
    if mode == "train":
        result["log"] = "训练模式"
        
        # 提取特征
        features = []
        for s in signal:
            csd = compute_cyclic_spectrum(s, fs, alpha_range[0], nfft)
            feat = extract_features(csd, alpha_range)
            features.append(feat)
        
        features = np.array(features)
        
        # 训练模型
        model = train_model(features, labels, model_path)
        result["result"] = "模型训练完成"
        
    elif mode in ["predict", "evaluate"]:
        result["log"] = "预测/评估模式"
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            result["log"] += " - 模型不存在，使用默认阈值检测"
            
            # 使用默认阈值检测
            detections = []
            for s in signal:
                # 计算循环谱密度
                csd = compute_cyclic_spectrum(s, fs, alpha_range[0], nfft)
                
                # 计算能量
                energy = np.sum(np.abs(csd)**2)
                
                # 使用阈值检测
                threshold = config.get("detection_threshold", 1e6)
                detection = 1 if energy > threshold else 0
                detections.append(detection)
            
            result["result"] = np.array(detections)
            
        else:
            # 加载模型
            model = load_model(model_path)
            
            # 提取特征
            features = []
            for s in signal:
                csd = compute_cyclic_spectrum(s, fs, alpha_range[0], nfft)
                feat = extract_features(csd, alpha_range)
                features.append(feat)
            
            features = np.array(features)
            
            # 预测
            predictions = model.predict(features)
            result["result"] = predictions
            
            # 如果是评估模式，计算性能指标
            if mode == "evaluate" and labels is not None:
                from sklearn.metrics import confusion_matrix
                
                # 计算混淆矩阵
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                
                # 计算性能指标
                detection_prob = tp / (tp + fn)
                false_alarm_prob = fp / (fp + tn)
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 保存性能指标
                result["metrics"] = {
                    "detection_probability": detection_prob,
                    "false_alarm_probability": false_alarm_prob,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
                
                result["log"] += f" - 评估完成，检测概率: {detection_prob:.4f}, 虚警概率: {false_alarm_prob:.4f}"
    
    return result
