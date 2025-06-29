import numpy as np
import pywt
import os
import json
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def _compute_wavelet_features(signal, wavelet, level):
    """计算改进的小波特征"""
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 1. 计算各层细节系数能量
    detail_energies = [np.sum(np.square(c)) for c in coeffs[1:]]
    
    # 2. 计算最大细节能量层
    max_detail_energy = np.max(detail_energies) if detail_energies else 0
    
    # 3. 计算小波熵
    total_energy = np.sum(np.square(coeffs[0])) + np.sum(detail_energies)
    normalized_energies = []
    
    # 近似系数能量
    approx_energy = np.sum(np.square(coeffs[0]))
    if total_energy > 0:
        normalized_energies.append(approx_energy / total_energy)
    
    # 细节系数能量
    for energy in detail_energies:
        if total_energy > 0:
            normalized_energies.append(energy / total_energy)
    
    # 计算熵
    entropy = 0
    for e in normalized_energies:
        if e > 0:
            entropy -= e * np.log(e)
    
    # 4. 计算能量比
    energy_ratio = np.sum(detail_energies) / total_energy if total_energy > 0 else 0
    
    # 5. 计算近似系数能量
    approx_energy_ratio = approx_energy / total_energy if total_energy > 0 else 0
    
    return [
        max_detail_energy,
        entropy,
        energy_ratio,
        approx_energy_ratio
    ]

def process(config: dict, signal: np.ndarray, labels: np.ndarray = None, mode: str = 'predict') -> dict:
    """
    小波变换频谱检测核心处理函数
    
    Args:
        config: 配置参数
        signal: 输入信号 (n_samples, signal_length)
        labels: 标签 (n_samples,)
        mode: 运行模式
        
    Returns:
        dict: 处理结果
    """
    # 初始化结果字典
    result = {}
    metrics = {}
    
    # 获取配置参数
    wavelet = config.get('wavelet', 'db4')
    level = config.get('level', 3)
    model_type = config.get('model_type', 'svm')
    
    # 处理模式判断
    if mode == 'train' and labels is None:
        raise ValueError("Labels are required for training mode")
    
    # 计算所有信号的小波特征
    features = []
    for sig in signal:
        # 处理复数信号（取模）
        if np.iscomplexobj(sig):
            sig = np.abs(sig)
        feat = _compute_wavelet_features(sig, wavelet, level)
        features.append(feat)
    
    features = np.array(features)
    
    # 根据不同模式处理
    if mode == 'train':
        # 训练模式：训练分类器
        if labels is None:
            raise ValueError("Labels are required for training")
        
        # 创建模型目录
        os.makedirs('models', exist_ok=True)
        
        # 使用SVM分类器
        if model_type == 'svm':
            clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 训练模型
        clf.fit(features, labels)
        
        # 保存模型
        model_path = os.path.join('models', 'wtd_model.pkl')
        joblib.dump(clf, model_path)
        
        # 计算训练集性能
        predictions = clf.predict(features)
        metrics = _calculate_metrics(predictions, labels)
        
        # 可视化特征分布
        _visualize_features(features, labels)
        
        result['result'] = "Model trained"
        result['log'] = f"Training completed. Model saved to {model_path}"
    
    elif mode in ['predict', 'evaluate']:
        # 预测或评估模式：加载模型
        model_path = os.path.join('models', 'wtd_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
        
        clf = joblib.load(model_path)
        
        # 进行预测
        predictions = clf.predict(features)
        result['result'] = predictions
        
        # 评估模式：计算性能指标
        if mode == 'evaluate':
            if labels is None:
                raise ValueError("Labels are required for evaluation")
            
            # 计算性能指标
            metrics = _calculate_metrics(predictions, labels)
            
            # 可视化特征分布
            _visualize_features(features, labels, mode='evaluate')
            
            result['log'] = "Evaluation completed"
        else:
            result['log'] = "Prediction completed"
    
    # 添加指标到结果
    result['metrics'] = metrics
    
    # 添加特征统计信息
    for i in range(features.shape[1]):
        result['metrics'][f'feat{i}_mean'] = np.mean(features[:, i])
        result['metrics'][f'feat{i}_std'] = np.std(features[:, i])
    
    return result

def _calculate_metrics(predictions, labels):
    """计算性能指标"""
    # 确保预测和标签长度一致
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")
    
    # 计算性能指标
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1_score': f1_score(labels, predictions, zero_division=0)
    }

def _visualize_features(features, labels, mode='train'):
    """可视化特征分布"""
    os.makedirs('assets', exist_ok=True)
    
    # 创建特征名称
    feature_names = ['Max Detail Energy', 'Wavelet Entropy', 'Energy Ratio', 'Approx Energy Ratio']
    
    # 创建3D散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制空闲和占用样本
    idle_mask = (labels == 0)
    occupied_mask = (labels == 1)
    
    ax.scatter(features[idle_mask, 0], features[idle_mask, 1], features[idle_mask, 2],
               c='blue', marker='o', alpha=0.5, label='Idle')
    ax.scatter(features[occupied_mask, 0], features[occupied_mask, 1], features[occupied_mask, 2],
               c='red', marker='^', alpha=0.7, label='Occupied')
    
    # 设置标签
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title('Wavelet Feature Distribution')
    ax.legend()
    
    # 保存图像
    plt.savefig('assets/feature_distribution.png', dpi=300)
    plt.close()