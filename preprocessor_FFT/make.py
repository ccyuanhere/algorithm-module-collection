import numpy as np
import os

def generate_test_signals(num_samples: int = 100, 
                         signal_length: int = 1024,
                         sampling_rate: float = 1000.0) -> np.ndarray:
    """
    生成用于FFT测试的信号数据
    
    Args:
        num_samples: 样本数量
        signal_length: 信号长度
        sampling_rate: 采样率(Hz)
        
    Returns:
        np.ndarray: 测试信号，形状为(num_samples, signal_length)
    """
    
    print(f"生成FFT测试信号数据...")
    print(f"样本数量: {num_samples}")
    print(f"信号长度: {signal_length}")
    print(f"采样率: {sampling_rate} Hz")
    
    t = np.linspace(0, signal_length/sampling_rate, signal_length, endpoint=False)
    signals = []
    
    for i in range(num_samples):
        # 生成不同类型的测试信号
        signal_type = np.random.choice(['sinusoid', 'multi_tone', 'chirp', 'pulse', 'noise'])
        
        if signal_type == 'sinusoid':
            # 单频正弦波
            frequency = np.random.uniform(10, 200)  # 10-200 Hz
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            
            signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            
        elif signal_type == 'multi_tone':
            # 多频正弦波叠加
            num_tones = np.random.randint(2, 5)
            signal = np.zeros(signal_length)
            
            for tone in range(num_tones):
                frequency = np.random.uniform(5, 250)
                amplitude = np.random.uniform(0.3, 1.0)
                phase = np.random.uniform(0, 2*np.pi)
                
                signal += amplitude * np.sin(2 * np.pi * frequency * t + phase)
                
        elif signal_type == 'chirp':
            # 线性调频信号（chirp）
            f0 = np.random.uniform(10, 50)   # 起始频率
            f1 = np.random.uniform(100, 300) # 结束频率
            amplitude = np.random.uniform(0.8, 1.5)
            
            # 线性调频
            frequency_sweep = f0 + (f1 - f0) * t / t[-1]
            phase = 2 * np.pi * np.cumsum(frequency_sweep) * (t[1] - t[0])
            
            signal = amplitude * np.sin(phase)
            
        elif signal_type == 'pulse':
            # 脉冲信号
            pulse_width = np.random.randint(10, 50)  # 脉冲宽度
            pulse_amplitude = np.random.uniform(1.0, 3.0)
            num_pulses = np.random.randint(3, 8)
            
            signal = np.zeros(signal_length)
            
            for pulse in range(num_pulses):
                start_idx = np.random.randint(0, signal_length - pulse_width)
                end_idx = start_idx + pulse_width
                
                # 矩形脉冲
                signal[start_idx:end_idx] = pulse_amplitude
                
        else:  # noise
            # 噪声信号
            noise_type = np.random.choice(['white', 'colored', 'impulse'])
            
            if noise_type == 'white':
                # 白噪声
                signal = np.random.normal(0, 1, signal_length)
                
            elif noise_type == 'colored':
                # 有色噪声（低通滤波白噪声）
                white_noise = np.random.normal(0, 1, signal_length)
                # 简单的移动平均滤波
                window_size = 5
                kernel = np.ones(window_size) / window_size
                signal = np.convolve(white_noise, kernel, mode='same')
                
            else:  # impulse
                # 脉冲噪声
                signal = np.random.normal(0, 0.1, signal_length)
                # 添加随机脉冲
                num_impulses = np.random.randint(5, 15)
                impulse_positions = np.random.choice(signal_length, num_impulses, replace=False)
                impulse_amplitudes = np.random.uniform(-3, 3, num_impulses)
                
                signal[impulse_positions] += impulse_amplitudes
        
        # 添加少量噪声
        noise_level = 0.05 * np.random.uniform(0.1, 0.5)
        noise = np.random.normal(0, noise_level, signal_length)
        signal = signal + noise
        
        signals.append(signal)
    
    return np.array(signals)

def generate_complex_signals(num_samples: int = 50,
                           signal_length: int = 1024,
                           sampling_rate: float = 1000.0) -> np.ndarray:
    """
    生成复数测试信号（I/Q信号）
    
    Args:
        num_samples: 样本数量
        signal_length: 信号长度
        sampling_rate: 采样率
        
    Returns:
        np.ndarray: 复数信号，形状为(num_samples, signal_length)
    """
    
    print(f"生成复数信号数据...")
    print(f"样本数量: {num_samples}")
    
    t = np.linspace(0, signal_length/sampling_rate, signal_length, endpoint=False)
    signals = []
    
    for i in range(num_samples):
        signal_type = np.random.choice(['qpsk', 'qam', 'fsk', 'complex_sinusoid'])
        
        # 初始化信号数组，确保长度正确
        signal = np.zeros(signal_length, dtype=complex)
        
        if signal_type == 'qpsk':
            # QPSK调制信号
            symbols_per_second = 50
            samples_per_symbol = int(sampling_rate / symbols_per_second)
            num_symbols = signal_length // samples_per_symbol + 1  # 多生成一个符号确保长度够
            
            # 生成随机比特
            bits = np.random.randint(0, 4, num_symbols)
            
            # QPSK映射
            qpsk_map = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
            symbols = np.array([qpsk_map[bit] for bit in bits])
            
            # 上采样
            upsampled = np.repeat(symbols, samples_per_symbol)
            # 确保不超过signal_length
            copy_length = min(len(upsampled), signal_length)
            signal[:copy_length] = upsampled[:copy_length]
            
        elif signal_type == 'qam':
            # 16-QAM调制信号
            constellation = np.array([
                -3-3j, -3-1j, -3+1j, -3+3j,
                -1-3j, -1-1j, -1+1j, -1+3j,
                1-3j,  1-1j,  1+1j,  1+3j,
                3-3j,  3-1j,  3+1j,  3+3j
            ]) / np.sqrt(10)  # 归一化
            
            symbols_per_second = 40
            samples_per_symbol = int(sampling_rate / symbols_per_second)
            num_symbols = signal_length // samples_per_symbol + 1  # 多生成一个符号
            
            # 随机选择星座点
            symbol_indices = np.random.randint(0, 16, num_symbols)
            symbols = constellation[symbol_indices]
            
            # 上采样
            upsampled = np.repeat(symbols, samples_per_symbol)
            # 确保不超过signal_length
            copy_length = min(len(upsampled), signal_length)
            signal[:copy_length] = upsampled[:copy_length]
            
        elif signal_type == 'fsk':
            # FSK信号
            f1 = 50  # 频率1
            f2 = 150 # 频率2
            bit_rate = 20
            samples_per_bit = int(sampling_rate / bit_rate)
            num_bits = signal_length // samples_per_bit + 1  # 多生成一个比特
            
            bits = np.random.randint(0, 2, num_bits)
            
            for bit_idx, bit in enumerate(bits):
                start_idx = bit_idx * samples_per_bit
                end_idx = min(start_idx + samples_per_bit, signal_length)
                
                if start_idx >= signal_length:
                    break
                    
                freq = f1 if bit == 0 else f2
                t_bit = t[start_idx:end_idx] - t[start_idx]
                signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_bit)
                
        else:  # complex_sinusoid
            # 复数正弦波
            frequency = np.random.uniform(10, 200)
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            
            signal = amplitude * np.exp(1j * (2 * np.pi * frequency * t + phase))
        
        # 添加复数噪声 - 现在signal长度肯定是signal_length
        noise_level = 0.05
        noise = (np.random.normal(0, noise_level, signal_length) + 
                1j * np.random.normal(0, noise_level, signal_length))
        
        # 现在两个数组长度都是signal_length，可以安全相加
        signal = signal + noise
        
        signals.append(signal)
    
    return np.array(signals)

def generate_example_data():
    """生成示例数据"""
    
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 参数设置
    signal_length = 1024
    sampling_rate = 1000.0
    
    print("=== 生成FFT测试数据 ===")
    
    # 1. 生成实数信号
    print("\n--- 生成实数信号 ---")
    real_signals = generate_test_signals(80, signal_length, sampling_rate)
    
    # 2. 生成复数信号
    print("\n--- 生成复数信号 ---")
    complex_signals = generate_complex_signals(50, signal_length, sampling_rate)
    
    # 3. 组合所有信号
    # 将复数信号转换为实部和虚部的形式以便存储
    complex_as_real = np.stack([complex_signals.real, complex_signals.imag], axis=1)
    
    # 为实数信号添加第二个通道（虚部为0）
    real_as_complex = np.stack([real_signals, np.zeros_like(real_signals)], axis=1)
    
    # 组合数据
    all_signals = np.concatenate([real_as_complex, complex_as_real], axis=0)
    
    print(f"\n=== 数据统计 ===")
    print(f"总信号数量: {all_signals.shape[0]}")
    print(f"信号形状: {all_signals.shape} (样本数, 通道数, 信号长度)")
    print(f"实数信号数量: {real_signals.shape[0]}")
    print(f"复数信号数量: {complex_signals.shape[0]}")
    print(f"数据范围: [{all_signals.min():.4f}, {all_signals.max():.4f}]")
    
    # 4. 生成对应的FFT输出
    print(f"\n--- 计算FFT输出 ---")
    
    outputs = []
    labels = []
    
    for i, signal_2ch in enumerate(all_signals):
        # 重新构造复数信号
        if i < len(real_signals):
            # 实数信号
            signal = signal_2ch[0]  # 只取实部
            fft_result = np.fft.fft(signal)
            label_type = 'real'
        else:
            # 复数信号
            signal = signal_2ch[0] + 1j * signal_2ch[1]
            fft_result = np.fft.fft(signal)
            label_type = 'complex'
        
        # 计算幅度谱
        magnitude_spectrum = np.abs(fft_result)
        outputs.append(magnitude_spectrum)
        labels.append({
            'type': label_type,
            'sampling_rate': sampling_rate,
            'signal_length': signal_length,
            'frequency_resolution': sampling_rate / signal_length
        })
    
    outputs = np.array(outputs)
    
    # 5. 保存数据
    np.save(os.path.join(data_dir, 'example_input.npy'), all_signals.astype(np.float32))
    np.save(os.path.join(data_dir, 'example_output.npy'), outputs.astype(np.float32))
    
    # 保存标签信息
    import json
    with open(os.path.join(data_dir, 'example_labels.json'), 'w') as f:
        json.dump(labels, f, indent=2)
    
    # 创建简化的标签数组
    label_array = np.array([1 if label['type'] == 'complex' else 0 for label in labels])
    np.save(os.path.join(data_dir, 'example_labels.npy'), label_array)
    
    print(f"\n=== 数据保存完成 ===")
    print(f"输入数据: {all_signals.shape} -> example_input.npy")
    print(f"输出数据: {outputs.shape} -> example_output.npy")
    print(f"标签数据: {len(labels)} -> example_labels.json")
    print(f"标签数组: {label_array.shape} -> example_labels.npy")
    print(f"采样率: {sampling_rate} Hz")
    print(f"信号长度: {signal_length} 采样点")
    print(f"频率分辨率: {sampling_rate/signal_length:.2f} Hz")
    print(f"数据保存到: {data_dir}")

if __name__ == "__main__":
    generate_example_data()
