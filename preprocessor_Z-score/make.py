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
    
    # 保存输入数据
    np.save(os.path.join(data_dir, 'example_input.npy'), all_signals.astype(np.float32))
    
    print(f"\n=== 数据保存完成 ===")
    print(f"输入数据: {all_signals.shape} -> example_input.npy")
    print(f"数据保存到: {data_dir}")
    print(f"注: Z-score标准化算法只需要输入数据，运行时会实时计算标准化结果")

if __name__ == "__main__":
    generate_example_data()
