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
    # 参数设置 - 增加挑战性
    N_total = 1000  # 总信号数量
    N_signal = 400  # 含目标信号的数量
    N_noise = N_total - N_signal  # 纯噪声信号数量
    
    signal_length = 512  # 信号长度
    symbol_rate = 1000  # 符号速率
    carrier_freq = 2000  # 载波频率
    sampling_freq = 8000  # 采样频率
    
    # 统一使用8位巴克码作为已知信号（与model.py中的匹配滤波器保持一致）
    # 注意：这里用0,1表示，对应model.py中的[1, -1, 1, 1, -1, -1, 1, -1]
    known_sequence = [1, 0, 1, 1, 0, 0, 1, 0]  # 8位巴克码
    
    signals = []
    labels = []
    
    print(f"生成 {N_signal} 个含目标信号的样本...")
    print(f"使用统一的已知序列: {known_sequence}")
    print(f"对应model.py中的匹配滤波器模板: {[2*bit-1 for bit in known_sequence]}")  # 转换为-1,1表示
    
    # 生成含目标信号的数据 - 增加难度
    for i in range(N_signal):
        # 随机选择信号类型和参数
        signal_type = np.random.choice(['bpsk', 'qpsk'])
        
        # 更严苛的信噪比范围：包含低SNR场景
        difficulty = np.random.choice(['easy', 'medium', 'hard'], p=[0.3, 0.4, 0.3])
        if difficulty == 'easy':
            snr_db = np.random.uniform(10, 15)
        elif difficulty == 'medium':
            snr_db = np.random.uniform(3, 8)
        else:  # hard
            snr_db = np.random.uniform(-2, 3)  # 包含负SNR
        
        if signal_type == 'bpsk':
            target_signal = generate_bpsk_signal(known_sequence, symbol_rate, carrier_freq, sampling_freq, snr_db)
        else:  # qpsk
            target_signal = generate_qpsk_signal(known_sequence, symbol_rate, carrier_freq, sampling_freq, snr_db)
        
        # 调整信号长度并增加挑战
        if len(target_signal) > signal_length:
            target_signal = target_signal[:signal_length]
        elif len(target_signal) < signal_length:
            # 在随机位置嵌入信号，使用更高的背景噪声
            start_pos = np.random.randint(0, signal_length - len(target_signal))
            
            # 根据难度调整背景噪声强度
            if difficulty == 'easy':
                bg_noise_std = 0.1
            elif difficulty == 'medium':
                bg_noise_std = 0.3
            else:  # hard
                bg_noise_std = 0.8  # 很强的背景噪声
            
            full_signal = np.random.normal(0, bg_noise_std, signal_length)
            
            # 有概率使用叠加而不是替换，模拟真实场景
            if np.random.random() < 0.5:
                full_signal[start_pos:start_pos+len(target_signal)] += target_signal  # 叠加
            else:
                full_signal[start_pos:start_pos+len(target_signal)] = target_signal  # 替换
            
            target_signal = full_signal
        
        signals.append(target_signal)
        labels.append(1)  # 有信号
    
    print(f"生成 {N_noise} 个纯噪声样本...")
    
    # 生成纯噪声数据 - 增加挑战性
    for i in range(N_noise):
        # 更多样化的噪声和干扰类型
        noise_type = np.random.choice(['gaussian', 'colored', 'interference', 'similar_signal', 'burst_noise'], 
                                     p=[0.2, 0.2, 0.2, 0.3, 0.1])
        
        if noise_type == 'gaussian':
            # 高斯白噪声 - 提高功率
            noise_signal = np.random.normal(0, np.random.uniform(0.8, 1.5), signal_length)
            
        elif noise_type == 'colored':
            # 有色噪声 - 更强
            white_noise = np.random.normal(0, np.random.uniform(1.0, 2.0), signal_length)
            b, a = sp_signal.butter(3, np.random.uniform(0.1, 0.5))  # 随机截止频率
            noise_signal = sp_signal.filtfilt(b, a, white_noise)
            
        elif noise_type == 'interference':
            # 干扰信号 - 使用相似参数但不同序列
            interfering_bits = np.random.randint(0, 2, 20)
            interfering_freq = carrier_freq + np.random.uniform(-200, 200)  # 更接近的频率
            interfering_rate = symbol_rate * np.random.uniform(0.8, 1.2)  # 相似的符号速率
            noise_signal = generate_bpsk_signal(interfering_bits, interfering_rate, interfering_freq, sampling_freq, 
                                               snr_db=np.random.uniform(5, 15))
            if len(noise_signal) > signal_length:
                noise_signal = noise_signal[:signal_length]
            elif len(noise_signal) < signal_length:
                noise_signal = np.pad(noise_signal, (0, signal_length - len(noise_signal)), 'constant')
                
        elif noise_type == 'similar_signal':
            # 相似但不完全匹配的信号 - 这是关键的挑战！
            similar_sequences = [
                [1, 0, 1, 1, 0, 0, 1, 1],  # 最后一位不同
                [1, 0, 1, 1, 0, 1, 1, 0],  # 中间几位不同
                [0, 0, 1, 1, 0, 0, 1, 0],  # 第一位不同
                [1, 0, 1, 1, 1, 0, 1, 0],  # 部分相似
            ]
            similar_seq = similar_sequences[np.random.randint(0, len(similar_sequences))]
            similar_snr = np.random.uniform(5, 15)  # 高SNR使其更容易造成虚警
            
            if np.random.random() < 0.5:
                noise_signal = generate_bpsk_signal(similar_seq, symbol_rate, carrier_freq, sampling_freq, similar_snr)
            else:
                noise_signal = generate_qpsk_signal(similar_seq, symbol_rate, carrier_freq, sampling_freq, similar_snr)
            
            # 嵌入到噪声背景中
            if len(noise_signal) < signal_length:
                start_pos = np.random.randint(0, signal_length - len(noise_signal))
                full_signal = np.random.normal(0, 0.3, signal_length)
                full_signal[start_pos:start_pos+len(noise_signal)] += noise_signal
                noise_signal = full_signal
            else:
                noise_signal = noise_signal[:signal_length]
                
        else:  # burst_noise
            # 突发噪声
            noise_signal = np.random.normal(0, 0.5, signal_length)
            # 添加随机突发
            num_bursts = np.random.randint(1, 4)
            for _ in range(num_bursts):
                burst_start = np.random.randint(0, signal_length - 50)
                burst_length = np.random.randint(20, 50)
                burst_end = min(burst_start + burst_length, signal_length)
                noise_signal[burst_start:burst_end] += np.random.normal(0, 2, burst_end - burst_start)
        
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
        f.write("- 目标信号包含BPSK和QPSK调制\n")
        f.write("- 挑战性测试数据，包含低SNR和强干扰场景\n")
        f.write("- 信噪比范围: -2 到 15 dB（包含负SNR）\n")
        f.write("- 背景噪声强度: 0.1 到 0.8（根据难度调整）\n")
        f.write("- 噪声类型: 高斯白噪声、有色噪声、干扰信号、相似信号、突发噪声\n")
        f.write("- 相似信号干扰: 使用部分匹配的序列造成虚警\n")
        f.write("- 统一使用8位巴克码作为已知序列: [1, 0, 1, 1, 0, 0, 1, 0]\n")
        f.write("- 对应匹配滤波器模板: [1, -1, 1, 1, -1, -1, 1, -1]\n")
        f.write("\n难度分级:\n")
        f.write("- 简单 (30%): SNR 10-15 dB, 低背景噪声\n")
        f.write("- 中等 (40%): SNR 3-8 dB, 中等背景噪声\n")
        f.write("- 困难 (30%): SNR -2-3 dB, 强背景噪声\n")
    
    print(f"数据描述已保存到: {description_file}")
    print("\n数据生成完成！")

if __name__ == "__main__":
    main()
