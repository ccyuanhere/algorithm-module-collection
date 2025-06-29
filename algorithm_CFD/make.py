import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_bpsk_signal(num_samples, symbol_rate, sampling_rate, snr_db, carrier_freq=0):
    """
    生成BPSK调制信号
    
    参数:
    num_samples: 样本数
    symbol_rate: 符号速率 (Hz)
    sampling_rate: 采样率 (Hz)
    snr_db: 信噪比 (dB)
    carrier_freq: 载波频率 (归一化频率)
    
    返回:
    modulated_signal: BPSK调制信号
    """
    # 生成随机比特序列
    num_symbols = int(num_samples * symbol_rate / sampling_rate)
    bits = np.random.randint(0, 2, num_symbols)
    
    # BPSK调制: 0-> -1, 1->1
    symbols = 2 * bits - 1
    
    # 上采样
    samples_per_symbol = int(sampling_rate / symbol_rate)
    upsampled = np.zeros(num_symbols * samples_per_symbol)
    upsampled[::samples_per_symbol] = symbols
    
    # 应用成型滤波器 (根升余弦) - 修复firwin参数问题
    # 使用正确的参数调用firwin
    rrc_filter = signal.firwin(
        101, 
        cutoff=0.5 * symbol_rate, 
        fs=sampling_rate, 
        window='hamming'
    )
    
    filtered = signal.convolve(upsampled, rrc_filter, mode='same')
    
    # 截断到所需长度
    filtered = filtered[:num_samples]
    
    # 添加载波
    t = np.arange(num_samples) / sampling_rate
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    modulated_signal = np.real(filtered * carrier)
    
    # 添加高斯白噪声
    signal_power = np.mean(modulated_signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
    noisy_signal = modulated_signal + noise
    
    return noisy_signal

def generate_test_data(num_segments=100, segment_length=2048, sampling_rate=8000):
    """
    生成测试数据集
    
    参数:
    num_segments: 信号段数量
    segment_length: 每段信号长度
    sampling_rate: 采样率
    
    返回:
    signals: 信号数组 [num_segments, segment_length]
    labels: 标签数组 [num_segments]
    """
    signals = np.zeros((num_segments, segment_length))
    labels = np.zeros(num_segments, dtype=int)
    
    # 信号参数
    symbol_rate = 1000  # 符号速率 (Hz)
    carrier_freq = 0.2  # 载波频率 (归一化)
    
    # 生成信号段
    for i in range(num_segments):
        # 随机决定是否有信号
        has_signal = np.random.rand() > 0.5
        labels[i] = 1 if has_signal else 0
        
        # 随机SNR (在-20dB到0dB之间)
        snr_db = np.random.uniform(-20, 0)
        
        if has_signal:
            # 生成带噪声的BPSK信号
            signals[i] = generate_bpsk_signal(
                segment_length, symbol_rate, sampling_rate, snr_db, carrier_freq
            )
        else:
            # 生成纯噪声
            signals[i] = np.random.normal(0, 1, segment_length)
    
    return signals, labels

if __name__ == "__main__":
    print("生成测试数据...")
    signals, labels = generate_test_data(num_segments=100, segment_length=2048)
    
    print("保存数据...")
    np.save("data/example_input.npy", signals)
    np.save("data/example_labels.npy", labels)
    
    print("生成示例输出...")
    # 为了完整性，生成一个示例输出（实际算法运行时会产生）
    np.save("data/example_output.npy", np.zeros(100, dtype=int))
    
    print("数据生成完成!")
    print(f"信号数据形状: {signals.shape}")
    print(f"标签数据形状: {labels.shape}")
    print(f"信号存在比例: {np.mean(labels)*100:.1f}%")