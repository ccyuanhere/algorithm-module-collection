import numpy as np
import os
from scipy import signal as sp_signal

def generate_bpsk_signal(bits, symbol_rate, carrier_freq, sampling_freq, snr_db=None):
    """
    生成BPSK调制信号
    
    Args:
        bits: 比特序列
        symbol_rate: 符号速率
        carrier_freq: 载波频率
        sampling_freq: 采样频率
        snr_db: 信噪比(dB)，如果为None则不添加噪声
    """
    samples_per_symbol = int(sampling_freq / symbol_rate)
    total_samples = len(bits) * samples_per_symbol
    
    # 时间轴
    t = np.arange(total_samples) / sampling_freq
    
    # 生成基带信号（NRZ编码）
    baseband = np.repeat(2 * np.array(bits) - 1, samples_per_symbol)  # 将0,1映射到-1,1
    
    # BPSK调制
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    modulated_signal = baseband * carrier
    
    # 添加噪声
    if snr_db is not None:
        signal_power = np.mean(modulated_signal**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(modulated_signal))
        modulated_signal += noise
    
    return modulated_signal

def generate_qpsk_signal(bits, symbol_rate, carrier_freq, sampling_freq, snr_db=None):
    """
    生成QPSK调制信号
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)  # 补零使长度为偶数
    
    samples_per_symbol = int(sampling_freq / symbol_rate)
    symbols = []
    
    # 将比特对映射到QPSK符号
    for i in range(0, len(bits), 2):
        bit_pair = (bits[i], bits[i+1])
        if bit_pair == (0, 0):
            symbols.append(1 + 1j)
        elif bit_pair == (0, 1):
            symbols.append(-1 + 1j)
        elif bit_pair == (1, 0):
            symbols.append(1 - 1j)
        else:  # (1, 1)
            symbols.append(-1 - 1j)
    
    # 上采样
    upsampled = np.repeat(symbols, samples_per_symbol)
    
    # 时间轴
    t = np.arange(len(upsampled)) / sampling_freq
    
    # 调制到载波
    I_signal = np.real(upsampled) * np.cos(2 * np.pi * carrier_freq * t)
    Q_signal = -np.imag(upsampled) * np.sin(2 * np.pi * carrier_freq * t)
    modulated_signal = I_signal + Q_signal
    
    # 添加噪声
    if snr_db is not None:
        signal_power = np.mean(modulated_signal**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(modulated_signal))
        modulated_signal += noise
    
    return modulated_signal

def generate_chirp_signal(duration, f_start, f_end, sampling_freq, snr_db=None):
    """
    生成线性调频信号
    """
    samples = int(duration * sampling_freq)
    t = np.linspace(0, duration, samples, endpoint=False)
    
    # 线性调频
    instantaneous_freq = f_start + (f_end - f_start) * t / duration
    signal = np.sin(2 * np.pi * np.cumsum(instantaneous_freq) / sampling_freq)
    
    # 添加噪声
    if snr_db is not None:
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        signal += noise
    
    return signal

def generate_matched_filter_test_data(config):
    """
    生成匹配滤波检测的测试数据
    """
    # 参数设置
    N_total = 1000  # 总信号数量
    N_signal = 400  # 含目标信号的数量
    N_noise = N_total - N_signal  # 纯噪声信号数量
    
    signal_length = 512  # 信号长度
    symbol_rate = 1000  # 符号速率
    carrier_freq = 2000  # 载波频率
    sampling_freq = 8000  # 采样频率
    
    # 已知信号序列（用于匹配滤波）
    known_sequences = [
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # 11位巴克码
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # 17位巴克码
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]  # 30位序列
    ]
    
    signals = []
    labels = []
    
    print(f"生成 {N_signal} 个含目标信号的样本...")
    
    # 生成含目标信号的数据
    for i in range(N_signal):
        # 随机选择信号类型和参数
        signal_type = np.random.choice(['bpsk', 'qpsk', 'chirp'])
        snr_db = np.random.uniform(0, 20)  # 信噪比范围
        
        if signal_type == 'bpsk':
            # 随机选择已知序列
            known_seq = known_sequences[np.random.randint(0, len(known_sequences))]
            target_signal = generate_bpsk_signal(known_seq, symbol_rate, carrier_freq, sampling_freq, snr_db)
        elif signal_type == 'qpsk':
            known_seq = known_sequences[0]  # 使用第一个序列
            target_signal = generate_qpsk_signal(known_seq, symbol_rate, carrier_freq, sampling_freq, snr_db)
        else:  # chirp
            duration = len(known_sequences[0]) / symbol_rate
            target_signal = generate_chirp_signal(duration, 1000, 3000, sampling_freq, snr_db)
        
        # 调整信号长度
        if len(target_signal) > signal_length:
            target_signal = target_signal[:signal_length]
        elif len(target_signal) < signal_length:
            # 在随机位置嵌入信号
            start_pos = np.random.randint(0, signal_length - len(target_signal))
            full_signal = np.random.normal(0, 0.1, signal_length)  # 低功率噪声背景
            full_signal[start_pos:start_pos+len(target_signal)] += target_signal
            target_signal = full_signal
        
        signals.append(target_signal)
        labels.append(1)  # 有信号
    
    print(f"生成 {N_noise} 个纯噪声样本...")
    
    # 生成纯噪声数据
    for i in range(N_noise):
        # 不同类型的噪声和干扰
        noise_type = np.random.choice(['gaussian', 'colored', 'interference'])
        
        if noise_type == 'gaussian':
            # 高斯白噪声
            noise_signal = np.random.normal(0, 1, signal_length)
        elif noise_type == 'colored':
            # 有色噪声
            white_noise = np.random.normal(0, 1, signal_length)
            # 通过低通滤波器产生有色噪声
            b, a = sp_signal.butter(3, 0.3)
            noise_signal = sp_signal.filtfilt(b, a, white_noise)
        else:  # interference
            # 干扰信号（不匹配的调制信号）
            interfering_bits = np.random.randint(0, 2, 20)
            interfering_freq = carrier_freq + np.random.uniform(-500, 500)
            noise_signal = generate_bpsk_signal(interfering_bits, symbol_rate*2, interfering_freq, sampling_freq, 
                                               snr_db=np.random.uniform(5, 15))
            if len(noise_signal) > signal_length:
                noise_signal = noise_signal[:signal_length]
            elif len(noise_signal) < signal_length:
                noise_signal = np.pad(noise_signal, (0, signal_length - len(noise_signal)), 'constant')
        
        signals.append(noise_signal)
        labels.append(0)  # 无信号
    
    # 转换为numpy数组
    signals = np.array(signals)
    labels = np.array(labels)
    
    # 打乱数据
    indices = np.random.permutation(N_total)
    signals = signals[indices]
    labels = labels[indices]
    
    print(f"数据生成完成:")
    print(f"  信号形状: {signals.shape}")
    print(f"  标签分布: {np.bincount(labels)}")
    print(f"  信号功率范围: {np.min(np.var(signals, axis=1)):.4f} - {np.max(np.var(signals, axis=1)):.4f}")
    
    return signals, labels

def main():
    """主函数：生成匹配滤波检测测试数据"""
    print("=== 生成匹配滤波检测测试数据 ===")
    
    # 读取配置
    config = {}
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 生成数据
    signals, labels = generate_matched_filter_test_data(config)
    
    # 保存数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    input_file = os.path.join(data_dir, 'example_input.npy')
    output_file = os.path.join(data_dir, 'example_output.npy')
    labels_file = os.path.join(data_dir, 'example_labels.npy')
    
    # 保存输入信号
    np.save(input_file, signals)
    print(f"输入信号已保存到: {input_file}")
    
    # 保存标签
    np.save(labels_file, labels)
    print(f"标签已保存到: {labels_file}")
    
    # 生成期望输出（简单的检测结果）
    expected_output = labels  # 期望输出就是标签本身
    np.save(output_file, expected_output)
    print(f"期望输出已保存到: {output_file}")
    
    # 生成数据描述文件
    description_file = os.path.join(data_dir, 'data_description.txt')
    with open(description_file, 'w', encoding='utf-8') as f:
        f.write("匹配滤波检测测试数据描述\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集大小: {len(signals)}\n")
        f.write(f"信号长度: {signals.shape[1]}\n")
        f.write(f"信号类型: 实数信号\n")
        f.write(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}\n")
        f.write(f"  - 0: 无目标信号\n")
        f.write(f"  - 1: 有目标信号\n\n")
        f.write("信号特点:\n")
        f.write("- 目标信号包含BPSK、QPSK和线性调频信号\n")
        f.write("- 噪声信号包含高斯白噪声、有色噪声和干扰信号\n")
        f.write("- 信噪比范围: 0-20 dB\n")
        f.write("- 使用巴克码作为已知序列\n")
    
    print(f"数据描述已保存到: {description_file}")
    print("\n数据生成完成！")

if __name__ == "__main__":
    main()
