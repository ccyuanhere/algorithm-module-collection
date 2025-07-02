import numpy as np
import os

def generate_test_signals(num_samples: int = 100, 
                         signal_length: int = 1000,
                         num_channels: int = 2) -> np.ndarray:
    """
    生成具有不同统计特性的测试信号数据
    
    Args:
        num_samples: 样本数量
        signal_length: 信号长度
        num_channels: 通道数
        
    Returns:
        np.ndarray: 测试信号，形状为(num_samples, num_channels, signal_length)
    """
    
    print(f"生成测试信号数据...")
    print(f"样本数量: {num_samples}")
    print(f"信号长度: {signal_length}")
    print(f"通道数: {num_channels}")
    
    signals = []
    
    for i in range(num_samples):
        # 生成不同统计特性的信号
        signal_type = np.random.choice(['normal', 'skewed', 'bimodal', 'uniform', 'exponential'])
        
        if signal_type == 'normal':
            # 正态分布信号（不同均值和方差）
            mean = np.random.uniform(-5, 5)
            std = np.random.uniform(0.5, 3.0)
            signal = np.random.normal(mean, std, (num_channels, signal_length))
            
        elif signal_type == 'skewed':
            # 偏态分布信号
            # 使用Gamma分布生成右偏数据
            shape = np.random.uniform(1, 5)
            scale = np.random.uniform(0.5, 2.0)
            signal = np.random.gamma(shape, scale, (num_channels, signal_length))
            
            # 随机决定是否左偏
            if np.random.random() < 0.5:
                signal = -signal
            
            # 添加位置偏移
            offset = np.random.uniform(-10, 10)
            signal += offset
            
        elif signal_type == 'bimodal':
            # 双峰分布信号
            # 两个正态分布的混合
            mean1 = np.random.uniform(-5, 0)
            mean2 = np.random.uniform(0, 5)
            std1 = np.random.uniform(0.5, 1.5)
            std2 = np.random.uniform(0.5, 1.5)
            
            # 混合比例
            mix_ratio = np.random.uniform(0.3, 0.7)
            n1 = int(signal_length * mix_ratio)
            n2 = signal_length - n1
            
            signal = np.zeros((num_channels, signal_length))
            for ch in range(num_channels):
                component1 = np.random.normal(mean1, std1, n1)
                component2 = np.random.normal(mean2, std2, n2)
                combined = np.concatenate([component1, component2])
                np.random.shuffle(combined)  # 随机混合
                signal[ch] = combined
                
        elif signal_type == 'uniform':
            # 均匀分布信号
            low = np.random.uniform(-10, 0)
            high = np.random.uniform(0, 10)
            signal = np.random.uniform(low, high, (num_channels, signal_length))
            
        else:  # exponential
            # 指数分布信号
            scale = np.random.uniform(0.5, 3.0)
            signal = np.random.exponential(scale, (num_channels, signal_length))
            
            # 随机添加负号和偏移
            if np.random.random() < 0.5:
                signal = -signal
            offset = np.random.uniform(-5, 5)
            signal += offset
        
        signals.append(signal)
    
    return np.array(signals)

def generate_outlier_signals(num_samples: int = 20,
                           signal_length: int = 1000,
                           num_channels: int = 2) -> np.ndarray:
    """
    生成包含异常值的测试信号
    """
    
    print(f"生成包含异常值的信号...")
    
    signals = []
    
    for i in range(num_samples):
        # 基础正态信号
        base_signal = np.random.normal(0, 1, (num_channels, signal_length))
        
        # 添加异常值
        outlier_ratio = np.random.uniform(0.01, 0.05)  # 1-5%的异常值
        num_outliers = int(signal_length * outlier_ratio)
        
        for ch in range(num_channels):
            # 随机选择异常值位置
            outlier_positions = np.random.choice(signal_length, num_outliers, replace=False)
            
            # 生成异常值（远离均值）
            outlier_values = np.random.choice([-1, 1], num_outliers) * np.random.uniform(5, 15, num_outliers)
            
            # 插入异常值
            base_signal[ch, outlier_positions] = outlier_values
        
        signals.append(base_signal)
    
    return np.array(signals)

def generate_extreme_signals(num_samples: int = 20,
                           signal_length: int = 1000,
                           num_channels: int = 2) -> np.ndarray:
    """
    生成极值测试信号
    """
    
    print(f"生成极值测试信号...")
    
    signals = []
    
    for i in range(num_samples):
        extreme_type = np.random.choice(['very_large', 'very_small', 'high_variance', 'zero_variance'])
        
        if extreme_type == 'very_large':
            # 非常大的均值
            mean = np.random.uniform(1000, 10000)
            std = np.random.uniform(10, 100)
            signal = np.random.normal(mean, std, (num_channels, signal_length))
            
        elif extreme_type == 'very_small':
            # 非常小的数值
            mean = np.random.uniform(-0.001, 0.001)
            std = np.random.uniform(0.0001, 0.001)
            signal = np.random.normal(mean, std, (num_channels, signal_length))
            
        elif extreme_type == 'high_variance':
            # 高方差信号
            mean = np.random.uniform(-1, 1)
            std = np.random.uniform(50, 200)
            signal = np.random.normal(mean, std, (num_channels, signal_length))
            
        else:  # zero_variance
            # 零方差信号（常数）
            constant_value = np.random.uniform(-10, 10)
            signal = np.full((num_channels, signal_length), constant_value)
            
            # 添加极少量噪声避免完全零方差
            noise = np.random.normal(0, 1e-10, (num_channels, signal_length))
            signal += noise
        
        signals.append(signal)
    
    return np.array(signals)

def generate_example_data():
    """生成示例数据"""
    
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 统一参数
    signal_length = 1000
    num_channels = 2
    
    # 1. 基本测试数据
    print("=== 生成基本测试数据 ===")
    basic_signals = generate_test_signals(60, signal_length, num_channels)
    
    # 2. 包含异常值的数据
    print("\n=== 生成异常值数据 ===")
    outlier_signals = generate_outlier_signals(20, signal_length, num_channels)
    
    # 3. 极值测试数据
    print("\n=== 生成极值测试数据 ===")
    extreme_signals = generate_extreme_signals(20, signal_length, num_channels)
    
    # 4. 组合所有数据
    all_signals = np.concatenate([basic_signals, outlier_signals, extreme_signals], axis=0)
    
    print(f"\n=== 数据统计 ===")
    print(f"总信号数量: {all_signals.shape[0]}")
    print(f"信号形状: {all_signals.shape}")
    print(f"数据范围: [{all_signals.min():.4f}, {all_signals.max():.4f}]")
    print(f"全局均值: {all_signals.mean():.4f}")
    print(f"全局标准差: {all_signals.std():.4f}")
    print(f"数据类型: {all_signals.dtype}")
    
    # 生成对应的标准化输出（使用不同方法）
    outputs = []
    labels = []
    
    for i, signal in enumerate(all_signals):
        # 随机选择标准化方法
        methods = ['standard', 'robust', 'quantile', 'clipped']
        method = methods[i % len(methods)]
        
        if method == 'standard':
            # 标准Z-score
            mean_val = signal.mean()
            std_val = signal.std()
            if std_val == 0:
                standardized = np.zeros_like(signal)
            else:
                standardized = (signal - mean_val) / std_val
                
        elif method == 'robust':
            # 鲁棒标准化（基于中位数和MAD）
            median_val = np.median(signal)
            mad = np.median(np.abs(signal - median_val))
            if mad == 0:
                standardized = np.zeros_like(signal)
            else:
                standardized = (signal - median_val) / (mad * 1.4826)
                
        elif method == 'quantile':
            # 分位数标准化
            q25 = np.percentile(signal, 25)
            q75 = np.percentile(signal, 75)
            iqr = q75 - q25
            if iqr == 0:
                standardized = np.zeros_like(signal)
            else:
                median_val = np.median(signal)
                standardized = (signal - median_val) / (iqr / 1.349)
                
        else:  # clipped
            # 裁剪异常值的标准化
            mean_val = signal.mean()
            std_val = signal.std()
            if std_val == 0:
                standardized = np.zeros_like(signal)
            else:
                standardized = (signal - mean_val) / std_val
                standardized = np.clip(standardized, -3, 3)  # 裁剪到±3σ
        
        outputs.append(standardized)
        labels.append(method)
    
    outputs = np.array(outputs)
    
    print(f"\n=== 标准化后统计 ===")
    print(f"标准化数据均值: {outputs.mean():.6f}")
    print(f"标准化数据标准差: {outputs.std():.6f}")
    print(f"标准化数据范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    # 保存数据
    np.save(os.path.join(data_dir, 'example_input.npy'), all_signals.astype(np.float32))
    np.save(os.path.join(data_dir, 'example_output.npy'), outputs.astype(np.float32))
    np.save(os.path.join(data_dir, 'example_labels.npy'), np.array(labels))
    
    print(f"\n=== 数据保存完成 ===")
    print(f"输入数据: {all_signals.shape} -> example_input.npy")
    print(f"输出数据: {outputs.shape} -> example_output.npy")
    print(f"标签数据: {len(labels)} -> example_labels.npy")
    print(f"数据保存到: {data_dir}")

if __name__ == "__main__":
    generate_example_data()
