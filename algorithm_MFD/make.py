import numpy as np
import os
from scipy import signal as sig
import json

def generate_bpsk_signal(bits, samples_per_bit=8, carrier_freq=2000, sampling_freq=8000):
    """
    生成BPSK调制的I/Q信号
    """
    # 基带信号（NRZ编码）
    baseband = np.repeat(2 * np.array(bits) - 1, samples_per_bit)  # 0,1 -> -1,1
    
    # 时间轴
    t = np.arange(len(baseband)) / sampling_freq
    
    # 生成I/Q信号（复数表示）
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    iq_signal = baseband * carrier
    
    return iq_signal

def generate_known_template():
    """
    生成已知信号模板（7位巴克码）
    """
    # 7位巴克码（标准序列）[1, 1, 1, -1, -1, 1, -1] 转换为 [1, 1, 1, 0, 0, 1, 0]
    barker_bits = [1, 1, 1, 0, 0, 1, 0]  # 7位巴克码（0/1格式）
    
    # 生成BPSK调制的I/Q模板
    template_iq = generate_bpsk_signal(barker_bits, samples_per_bit=8)
    
    return np.array(barker_bits), template_iq

def create_iq_array(complex_signal):
    """
    将复数信号转换为I/Q数组格式
    """
    return np.column_stack([np.real(complex_signal), np.imag(complex_signal)])

def generate_example_data(num_samples=1000, signal_length=512, snr_range=(-2, 15), 
                         carrier_freq=2000, sampling_freq=8000):
    """
    生成匹配滤波检测的I/Q测试数据
    
    Args:
        num_samples: 样本数量
        signal_length: 每个信号长度（I/Q采样点数）
        snr_range: 信噪比范围(dB)
        carrier_freq: 载波频率(Hz)
        sampling_freq: 采样频率(Hz)
    """
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成已知信号模板
    template_bits, template_iq = generate_known_template()
    template_length = len(template_iq)
    
    print(f"生成匹配滤波检测I/Q测试数据...")
    print(f"已知模板: {template_bits}")
    print(f"模板长度: {template_length} I/Q采样点")
    
    # 初始化数据数组
    # 信号数据：(样本数, 信号长度, 2) - 最后一维是[I, Q]
    signal_data = np.zeros((num_samples, signal_length, 2))
    labels = np.zeros(num_samples, dtype=int)
    
    # 生成信号样本
    for i in range(num_samples):
        # 随机决定是否包含目标信号 (50%概率)
        has_target = np.random.random() < 0.5
        labels[i] = 1 if has_target else 0
        
        # 随机选择SNR
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        
        if has_target:
            # 生成含目标信号的I/Q数据
            signal_iq = generate_target_signal(template_iq, signal_length, snr_db, 
                                             carrier_freq, sampling_freq)
        else:
            # 生成干扰和噪声信号
            signal_iq = generate_interference_signal(template_iq, signal_length, snr_db,
                                                   carrier_freq, sampling_freq)
        
        # 转换为I/Q数组格式
        signal_data[i] = create_iq_array(signal_iq)
    
    # 保存I/Q信号数据
    np.save(os.path.join(data_dir, 'example_input.npy'), signal_data)
    np.save(os.path.join(data_dir, 'example_labels.npy'), labels)
    
    # 保存已知模板（供算法使用）
    template_iq_array = create_iq_array(template_iq)
    np.save(os.path.join(data_dir, 'known_template.npy'), template_iq_array)
    
    # 保存元数据
    metadata = {
        'num_samples': num_samples,
        'signal_length': signal_length,
        'template_length': template_length,
        'template_bits': template_bits.tolist(),
        'snr_range': snr_range,
        'carrier_freq': carrier_freq,
        'sampling_freq': sampling_freq,
        'signal_format': 'I/Q',
        'data_shape': signal_data.shape
    }
    
    with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"数据生成完成:")
    print(f"  样本数量: {num_samples}")
    print(f"  信号长度: {signal_length} I/Q采样点")
    print(f"  数据形状: {signal_data.shape} (样本数, 时间点, I/Q)")
    print(f"  标签分布: {np.bincount(labels)}")
    print(f"  SNR范围: {snr_range[0]} - {snr_range[1]} dB")
    print(f"  数据已保存到: {data_dir}")

def generate_target_signal(template_iq, signal_length, snr_db, carrier_freq, sampling_freq):
    """
    生成包含目标信号的I/Q序列
    """
    # 创建噪声背景
    noise_iq = (np.random.normal(0, 1, signal_length) + 
                1j * np.random.normal(0, 1, signal_length))
    
    # 在随机位置嵌入目标信号
    if len(template_iq) < signal_length:
        start_pos = np.random.randint(0, signal_length - len(template_iq))
        signal_iq = noise_iq.copy()
        
        # 计算信号功率并调整SNR
        signal_power = np.mean(np.abs(template_iq)**2)
        noise_power = np.mean(np.abs(noise_iq)**2)
        
        # 调整信号幅度以达到目标SNR
        target_signal_power = noise_power * (10**(snr_db/10))
        scale_factor = np.sqrt(target_signal_power / signal_power)
        
        # 嵌入缩放后的目标信号
        signal_iq[start_pos:start_pos+len(template_iq)] += scale_factor * template_iq
    else:
        # 如果模板长度 >= 信号长度，直接使用模板的前部分
        signal_iq = template_iq[:signal_length] + noise_iq
    
    return signal_iq

def generate_interference_signal(template_iq, signal_length, snr_db, carrier_freq, sampling_freq):
    """
    生成干扰信号（无目标信号）
    """
    interference_type = np.random.choice(['noise', 'similar_signal', 'other_modulation'])
    
    if interference_type == 'noise':
        # 纯噪声
        signal_iq = (np.random.normal(0, 1, signal_length) + 
                     1j * np.random.normal(0, 1, signal_length))
    
    elif interference_type == 'similar_signal':
        # 相似但不同的信号（制造虚警）
        # 使用不同的比特序列
        fake_bits = np.random.randint(0, 2, 8)  # 随机8位
        fake_template_iq = generate_bpsk_signal(fake_bits, samples_per_bit=8)
        
        # 嵌入假信号
        noise_iq = (np.random.normal(0, 1, signal_length) + 
                    1j * np.random.normal(0, 1, signal_length))
        
        if len(fake_template_iq) < signal_length:
            start_pos = np.random.randint(0, signal_length - len(fake_template_iq))
            signal_iq = noise_iq.copy()
            
            # 稍微降低假信号的功率
            scale_factor = 0.7 * np.sqrt(10**(snr_db/10))
            signal_iq[start_pos:start_pos+len(fake_template_iq)] += scale_factor * fake_template_iq
        else:
            signal_iq = fake_template_iq[:signal_length] + noise_iq
    
    else:  # other_modulation
        # 其他调制方式的干扰
        # 生成QPSK信号作为干扰
        random_bits = np.random.randint(0, 2, 16)  # 16位随机序列
        
        # QPSK调制
        qpsk_symbols = []
        for i in range(0, len(random_bits), 2):
            bit_pair = (random_bits[i], random_bits[i+1] if i+1 < len(random_bits) else 0)
            if bit_pair == (0, 0):
                qpsk_symbols.append(1 + 1j)
            elif bit_pair == (0, 1):
                qpsk_symbols.append(-1 + 1j)
            elif bit_pair == (1, 0):
                qpsk_symbols.append(1 - 1j)
            else:
                qpsk_symbols.append(-1 - 1j)
        
        # 上采样
        samples_per_symbol = 8
        qpsk_signal = np.repeat(qpsk_symbols, samples_per_symbol)
        
        # 添加载波
        t = np.arange(len(qpsk_signal)) / sampling_freq
        carrier_offset = carrier_freq + np.random.uniform(-500, 500)  # 频率偏移
        carrier = np.exp(1j * 2 * np.pi * carrier_offset * t)
        qpsk_iq = qpsk_signal * carrier
        
        # 嵌入到噪声中
        noise_iq = (np.random.normal(0, 1, signal_length) + 
                    1j * np.random.normal(0, 1, signal_length))
        
        if len(qpsk_iq) < signal_length:
            start_pos = np.random.randint(0, signal_length - len(qpsk_iq))
            signal_iq = noise_iq.copy()
            scale_factor = 0.8 * np.sqrt(10**(snr_db/10))
            signal_iq[start_pos:start_pos+len(qpsk_iq)] += scale_factor * qpsk_iq
        else:
            signal_iq = qpsk_iq[:signal_length] + noise_iq
    
    return signal_iq

if __name__ == "__main__":
    # 从配置文件加载参数
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 生成示例数据
    generate_example_data(
        num_samples=config.get('num_samples', 1000),
        signal_length=config.get('signal_length', 512),
        snr_range=config.get('snr_range', [-2, 15]),
        carrier_freq=config.get('carrier_freq', 2000),
        sampling_freq=config.get('sampling_freq', 8000)
    )
