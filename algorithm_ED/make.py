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
        np.ndarray: 生成的信号，形状为(signal_length, 2) [I, Q]
    """
    # 生成基础噪声
    noise = np.random.normal(0, 1, signal_length) + 1j * np.random.normal(0, 1, signal_length)
    
    if has_primary_user:
        # 生成更容易检测的主用户信号
        t = np.arange(signal_length)
        
        # 随机选择信号类型，增加多样性
        signal_type = np.random.choice(['sinusoid', 'qpsk', 'multi_tone', 'chirp'])
        
        if signal_type == 'sinusoid':
            # 单频正弦波（增强版）
            carrier_freq = np.random.uniform(0.05, 0.2)  # 随机载波频率
            phase = np.random.uniform(0, 2*np.pi)
            primary_signal = np.exp(1j * (2 * np.pi * carrier_freq * t + phase))
            
        elif signal_type == 'qpsk':
            # QPSK信号（符号级变化）
            symbol_rate = 100  # 符号率
            samples_per_symbol = max(1, signal_length // symbol_rate)  # 确保至少为1
            
            # 计算需要的符号数，确保能覆盖整个信号长度
            num_symbols = (signal_length + samples_per_symbol - 1) // samples_per_symbol  # 向上取整
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
            
            # 生成QPSK信号，确保长度正确
            primary_signal = np.repeat(symbols, samples_per_symbol)[:signal_length]
            
            # 如果长度仍然不够，用最后一个符号填充
            if len(primary_signal) < signal_length:
                padding_length = signal_length - len(primary_signal)
                padding = np.full(padding_length, symbols[-1])
                primary_signal = np.concatenate([primary_signal, padding])
            
        elif signal_type == 'multi_tone':
            # 多音调信号
            num_tones = np.random.randint(2, 5)
            primary_signal = np.zeros(signal_length, dtype=complex)
            
            for _ in range(num_tones):
                freq = np.random.uniform(0.05, 0.25)
                amp = np.random.uniform(0.5, 1.5)
                phase = np.random.uniform(0, 2*np.pi)
                primary_signal += amp * np.exp(1j * (2 * np.pi * freq * t + phase))
                
        else:  # chirp
            # 线性调频信号
            f0 = 0.05
            f1 = 0.25
            primary_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * signal_length)))
        
        # 确保primary_signal长度正确
        assert len(primary_signal) == signal_length, f"Signal length mismatch: {len(primary_signal)} != {signal_length}"
        
        # 调整信号功率以匹配指定的SNR
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(np.abs(primary_signal) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)
        
        # 确保信号功率足够强
        min_signal_power = noise_power * max(snr_linear, 2.0)  # 至少3dB SNR
        target_signal_power = max(min_signal_power, noise_power * snr_linear)
        
        scaling_factor = np.sqrt(target_signal_power / signal_power)
        
        # 组合信号和噪声
        signal = primary_signal * scaling_factor + noise
        
    else:
        # 纯噪声信号（减少干扰脉冲的强度和数量）
        num_pulses = np.random.randint(1, 4)  # 减少脉冲数量：1-3个
        
        if num_pulses > 0:
            pulse_positions = np.random.choice(signal_length, num_pulses, replace=False)
            pulse_amplitudes = np.random.uniform(3, 8, num_pulses)  # 减少脉冲强度：3-8倍
            
            # 添加较温和的干扰脉冲
            for i in range(num_pulses):
                pos = pulse_positions[i]
                amp = pulse_amplitudes[i]
                # 减少脉冲持续时间：5个点
                start = max(0, pos-2)
                end = min(signal_length, pos+3)
                for p in range(start, end):
                    noise[p] += amp * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
        
        signal = noise
    
    # 转换为实部和虚部的二维数组
    signal_real_imag = np.column_stack((np.real(signal), np.imag(signal)))
    
    return signal_real_imag

def generate_example_data(num_samples: int = 100, signal_length: int = 8192, snr_range: Tuple[float, float] = (-5, 15)) -> None:
    """
    生成示例数据
    
    Args:
        num_samples: 样本数量
        signal_length: 信号长度
        snr_range: 信噪比范围(dB) - 调整为更高的SNR范围
    """
    print(f"生成能量检测算法测试数据...")
    print(f"样本数量: {num_samples}")
    print(f"信号长度: {signal_length}")
    print(f"SNR范围: {snr_range[0]} ~ {snr_range[1]} dB")
    
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成输入信号和标签
    inputs = []
    labels = []
    
    for i in range(num_samples):
        # 随机决定是否有主用户
        has_primary_user = np.random.choice([True, False])
        
        # 随机选择SNR（调整分布，偏向更高SNR）
        if has_primary_user:
            # 对于有信号的情况，使用稍高的SNR
            snr_db = np.random.uniform(snr_range[0] + 2, snr_range[1])
        else:
            # 对于无信号的情况，SNR不适用，但保持一致性
            snr_db = np.random.uniform(snr_range[0], snr_range[1])
        
        # 生成信号
        signal = generate_signal(signal_length, snr_db, has_primary_user)
        
        inputs.append(signal)
        labels.append(1 if has_primary_user else 0)
        
        # 进度显示
        if (i + 1) % 20 == 0 or (i + 1) == num_samples:
            print(f"已生成: {i + 1}/{num_samples} 个样本")
    
    # 转换为numpy数组
    inputs = np.array(inputs)
    labels = np.array(labels)
    
    # 显示数据统计
    print(f"\n=== 数据统计 ===")
    print(f"输入数据形状: {inputs.shape}")
    print(f"标签分布: 有信号={np.sum(labels)}, 无信号={len(labels) - np.sum(labels)}")
    print(f"信号占比: {np.mean(labels):.1%}")
    
    # 保存数据
    np.save(os.path.join(data_dir, 'example_input.npy'), inputs.astype(np.float32))
    np.save(os.path.join(data_dir, 'example_labels.npy'), labels.astype(np.int32))
    
    print(f"\n数据已保存到: {data_dir}")
    print(f"- example_input.npy: 输入信号数据")
    print(f"- example_labels.npy: 标签数据")

if __name__ == "__main__":
    generate_example_data()