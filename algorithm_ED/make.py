import numpy as np
import os
from typing import Tuple

def generate_signal(signal_length: int, snr_db: float, has_primary_user: bool = False) -> np.ndarray:
    """
    生成测试信号
    
    Args:
        signal_length: 信号长度
        snr_db: 信噪比(dB)
        has_primary_user: 是否包含主用户信号
        
    Returns:
        np.ndarray: 生成的信号
    """
    # 生成噪声
    noise = np.random.normal(0, 1, signal_length) + 1j * np.random.normal(0, 1, signal_length)
    
    if has_primary_user:
        # 生成主用户信号（这里简化为正弦波）
        t = np.arange(signal_length)
        carrier_freq = 0.1  # 载波频率
        primary_signal = np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # 调整信号功率以匹配指定的SNR
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(np.abs(primary_signal) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)
        scaling_factor = np.sqrt(snr_linear * noise_power / signal_power)
        
        # 组合信号和噪声
        signal = primary_signal * scaling_factor + noise
    else:
        # 只有噪声
        signal = noise
    
    # 转换为实部和虚部的二维数组
    signal_real_imag = np.column_stack((np.real(signal), np.imag(signal)))
    
    return signal_real_imag

def generate_example_data(num_samples: int = 100, signal_length: int = 8192, snr_range: Tuple[float, float] = (-10, 20)) -> None:
    """
    生成示例数据
    
    Args:
        num_samples: 样本数量
        signal_length: 信号长度
        snr_range: 信噪比范围(dB)
    """
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成输入信号和标签
    inputs = []
    labels = []
    
    for i in range(num_samples):
        # 随机决定是否有主用户
        has_primary_user = np.random.choice([True, False])
        
        # 随机选择SNR
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        
        # 生成信号
        signal = generate_signal(signal_length, snr_db, has_primary_user)
        
        inputs.append(signal)
        labels.append(1 if has_primary_user else 0)
    
    # 转换为numpy数组
    inputs = np.array(inputs)
    labels = np.array(labels)
    
    # 计算输出（这里使用简化的能量检测作为参考）
    outputs = []
    window_size = 1024
    threshold_factor = 1.5
    
    for signal in inputs:
        # 转换为复数形式
        signal_complex = signal[:, 0] + 1j * signal[:, 1]
        
        # 计算能量
        energy = np.zeros(signal_length // window_size)
        for j in range(0, signal_length - window_size + 1, window_size):
            window = signal_complex[j:j + window_size]
            energy[j // window_size] = np.sum(np.abs(window) ** 2)
        
        # 计算阈值
        threshold = np.mean(energy) * threshold_factor
        
        # 决策
        decision = energy > threshold
        outputs.append(decision)
    
    outputs = np.array(outputs)
    
    # 保存数据
    np.save(os.path.join(data_dir, 'example_input.npy'), inputs)
    np.save(os.path.join(data_dir, 'example_output.npy'), outputs)
    np.save(os.path.join(data_dir, 'example_labels.npy'), labels)
    
    print(f"示例数据已生成并保存到 {data_dir}")

if __name__ == "__main__":
    generate_example_data()
