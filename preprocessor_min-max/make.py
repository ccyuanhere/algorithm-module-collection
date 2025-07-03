import numpy as np
import os

def generate_test_signals(num_samples: int = 100, 
                         signal_length: int = 1000,
                         num_channels: int = 2) -> np.ndarray:
    """
    生成测试信号数据
    
    Args:
        num_samples: 样本数量
        signal_length: 信号长度
        num_channels: 通道数（如I/Q两通道）
        
    Returns:
        np.ndarray: 测试信号，形状为(num_samples, num_channels, signal_length)
    """
    
    print(f"生成测试信号数据...")
    print(f"样本数量: {num_samples}")
    print(f"信号长度: {signal_length}")
    print(f"通道数: {num_channels}")
    
    signals = []
    
    for i in range(num_samples):
        # 生成不同类型的测试信号
        signal_type = np.random.choice(['sinusoid', 'chirp', 'noise', 'mixed'])
        
        if signal_type == 'sinusoid':
            # 正弦波信号
            t = np.linspace(0, 10, signal_length)
            freq = np.random.uniform(0.5, 5.0)
            amplitude = np.random.uniform(0.5, 10.0)
            phase = np.random.uniform(0, 2*np.pi)
            
            signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # 为每个通道添加不同的相位偏移
            channels = []
            for ch in range(num_channels):
                ch_phase = phase + ch * np.pi / 2  # 90度相位差
                ch_signal = amplitude * np.sin(2 * np.pi * freq * t + ch_phase)
                channels.append(ch_signal)
            
            signal = np.array(channels)
            
        elif signal_type == 'chirp':
            # 线性调频信号
            t = np.linspace(0, 5, signal_length)
            f0 = np.random.uniform(0.1, 1.0)
            f1 = np.random.uniform(2.0, 5.0)
            amplitude = np.random.uniform(1.0, 8.0)
            
            # 线性调频
            instantaneous_freq = f0 + (f1 - f0) * t / t[-1]
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) * (t[1] - t[0])
            
            channels = []
            for ch in range(num_channels):
                ch_phase = phase + ch * np.pi / 4
                ch_signal = amplitude * np.sin(ch_phase)
                channels.append(ch_signal)
            
            signal = np.array(channels)
            
        elif signal_type == 'noise':
            # 噪声信号
            noise_type = np.random.choice(['gaussian', 'uniform', 'exponential'])
            
            if noise_type == 'gaussian':
                mean = np.random.uniform(-2, 2)
                std = np.random.uniform(0.5, 3.0)
                signal = np.random.normal(mean, std, (num_channels, signal_length))
            elif noise_type == 'uniform':
                low = np.random.uniform(-10, 0)
                high = np.random.uniform(0, 10)
                signal = np.random.uniform(low, high, (num_channels, signal_length))
            else:  # exponential
                scale = np.random.uniform(0.5, 3.0)
                signal = np.random.exponential(scale, (num_channels, signal_length))
                # 随机添加负号
                signal = signal * np.random.choice([-1, 1], (num_channels, signal_length))
                
        else:  # mixed
            # 混合信号
            t = np.linspace(0, 8, signal_length)
            
            # 多个正弦波叠加
            num_components = np.random.randint(2, 5)
            signal = np.zeros((num_channels, signal_length))
            
            for comp in range(num_components):
                freq = np.random.uniform(0.2, 3.0)
                amplitude = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2*np.pi)
                
                for ch in range(num_channels):
                    ch_phase = phase + ch * np.pi / 3
                    signal[ch] += amplitude * np.sin(2 * np.pi * freq * t + ch_phase)
            
            # 添加噪声
            noise_level = np.random.uniform(0.1, 0.5)
            noise = np.random.normal(0, noise_level, (num_channels, signal_length))
            signal += noise
            
        signals.append(signal)
    
    return np.array(signals)

def generate_example_data():
    """生成示例数据"""
    
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成不同类型的测试数据
    
    # 统一信号长度
    signal_length = 1000
    num_channels = 2
    
    # 1. 基本测试数据
    print("=== 生成基本测试数据 ===")
    basic_signals = generate_test_signals(50, signal_length, num_channels)
    
    # 2. 多通道数据（取前2个通道保持一致）
    print("\n=== 生成多通道数据 ===")
    multichannel_signals = generate_test_signals(30, signal_length, 4)
    multichannel_signals = multichannel_signals[:, :num_channels, :]  # 只取前2个通道
    
    # 3. 极值测试数据
    print("\n=== 生成极值测试数据 ===")
    extreme_signals = []
    
    # 很大的值
    large_signal = np.random.uniform(1000, 10000, (10, num_channels, signal_length))
    extreme_signals.append(large_signal)
    
    # 很小的值
    small_signal = np.random.uniform(-0.001, 0.001, (10, num_channels, signal_length))
    extreme_signals.append(small_signal)
    
    # 混合极值
    mixed_extreme = np.random.uniform(-5000, 5000, (10, num_channels, signal_length))
    extreme_signals.append(mixed_extreme)
    
    extreme_signals = np.concatenate(extreme_signals, axis=0)
    
    # 4. 组合所有数据（现在所有数组的形状都是 (N, 2, 1000)）
    all_signals = np.concatenate([basic_signals, multichannel_signals, extreme_signals], axis=0)
    
    print(f"\n=== 数据统计 ===")
    print(f"总信号数量: {all_signals.shape[0]}")
    print(f"信号形状: {all_signals.shape}")
    print(f"数据范围: [{all_signals.min():.4f}, {all_signals.max():.4f}]")
    print(f"数据类型: {all_signals.dtype}")
    
    # 生成对应的归一化输出（使用不同归一化范围）
    outputs = []
    labels = []
    
    for i, signal in enumerate(all_signals):
        # 随机选择归一化范围
        norm_ranges = [(0, 1), (-1, 1), (0, 10), (-5, 5)]
        norm_range = norm_ranges[i % len(norm_ranges)]
        
        # 执行归一化
        min_val = signal.min()
        max_val = signal.max()
        range_val = max_val - min_val
        
        if range_val == 0:
            normalized = np.full_like(signal, (norm_range[0] + norm_range[1]) / 2)
        else:
            # 标准min-max归一化
            normalized_01 = (signal - min_val) / range_val
            normalized = normalized_01 * (norm_range[1] - norm_range[0]) + norm_range[0]
        
        outputs.append(normalized)
        labels.append(norm_range)
    
    # 保存数据（只保存输入数据）
    np.save(os.path.join(data_dir, 'example_input.npy'), all_signals.astype(np.float32))
    
    print(f"\n=== 数据保存完成 ===")
    print(f"输入数据: {all_signals.shape} -> example_input.npy")
    print(f"数据保存到: {data_dir}")
    print(f"数据包含 {all_signals.shape[0]} 个信号样本，用于测试归一化算法效果")

if __name__ == "__main__":
    generate_example_data()
