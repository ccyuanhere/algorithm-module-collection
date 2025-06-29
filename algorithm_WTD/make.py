import numpy as np
import os
import matplotlib.pyplot as plt

def generate_signal(signal_length, snr_db, occupied=True):
    """
    生成测试信号
    
    Args:
        signal_length: 信号长度
        snr_db: 信噪比 (dB)
        occupied: 是否被占用
        
    Returns:
        complex signal
    """
    # 生成噪声
    noise = (np.random.normal(0, 1, signal_length) + 
             1j * np.random.normal(0, 1, signal_length)) / np.sqrt(2)
    
    if not occupied:
        return noise
    
    # 随机选择调制类型
    mod_type = np.random.choice(['qpsk', 'ofdm', 'am'], p=[0.4, 0.4, 0.2])
    
    if mod_type == 'qpsk':
        # QPSK调制信号
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], signal_length)
        signal = symbols
    
    elif mod_type == 'ofdm':
        # OFDM信号
        n_subcarriers = 64
        n_symbols = (signal_length // n_subcarriers) + 1
        
        # 生成OFDM符号
        ofdm_symbols = []
        for _ in range(n_symbols):
            # 随机QAM符号
            qam_symbols = (np.random.randint(0, 4, n_subcarriers) * 2 - 3) + 1j*(np.random.randint(0, 4, n_subcarriers) * 2 - 3)
            # IFFT变换
            symbol = np.fft.ifft(qam_symbols)
            # 添加循环前缀
            cp = symbol[-n_subcarriers//4:]
            ofdm_symbols.extend(cp)
            ofdm_symbols.extend(symbol)
        
        # 截取到所需长度
        signal = np.array(ofdm_symbols[:signal_length])
    
    else:  # am
        # AM调制信号
        t = np.arange(signal_length)
        carrier_freq = 0.2 + np.random.uniform(-0.05, 0.05)
        mod_freq = 0.01 + np.random.uniform(-0.005, 0.005)
        
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        modulation = 0.5 * np.sin(2 * np.pi * mod_freq * t) + 0.5
        signal = modulation * carrier
    
    # 调整信号功率以满足SNR要求
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = np.mean(np.abs(noise)**2)
    
    # 计算所需缩放因子
    target_snr = 10**(snr_db / 10)
    scale = np.sqrt(target_snr * noise_power / signal_power)
    
    return scale * signal + noise

def generate_data():
    """生成测试数据并保存"""
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 设置参数
    n_samples = 500
    signal_length = 1024
    snr_range = [-10, -5, 0]  # 多种SNR环境
    
    # 生成空闲信号 (50%)
    idle_signals = []
    for _ in range(n_samples // 2):
        snr_db = np.random.choice(snr_range)
        idle_signals.append(generate_signal(signal_length, snr_db, occupied=False))
    
    # 生成占用信号 (50%)
    occupied_signals = []
    for _ in range(n_samples // 2):
        snr_db = np.random.choice(snr_range)
        occupied_signals.append(generate_signal(signal_length, snr_db, occupied=True))
    
    # 合并信号
    signals = np.array(idle_signals + occupied_signals)
    
    # 生成标签 (0: 空闲, 1: 占用)
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # 保存数据
    np.save('data/example_input.npy', signals)
    np.save('data/example_labels.npy', labels)
    
    print(f"Generated {n_samples} signals saved to data/ directory")
    print(f"  - Idle signals: {len(idle_signals)}")
    print(f"  - Occupied signals: {len(occupied_signals)}")
    print(f"  - SNR range: {snr_range} dB")
    
    # 可视化示例信号
    os.makedirs('assets', exist_ok=True)
    
    # 空闲信号示例
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(np.real(idle_signals[0]))
    plt.title('Idle Signal (Real Part)')
    
    plt.subplot(2, 2, 2)
    plt.plot(np.imag(idle_signals[0]))
    plt.title('Idle Signal (Imaginary Part)')
    
    # 占用信号示例
    plt.subplot(2, 2, 3)
    plt.plot(np.real(occupied_signals[0]))
    plt.title('Occupied Signal (Real Part)')
    
    plt.subplot(2, 2, 4)
    plt.plot(np.imag(occupied_signals[0]))
    plt.title('Occupied Signal (Imaginary Part)')
    
    plt.tight_layout()
    plt.savefig('assets/signal_examples.png', dpi=300)
    plt.close()
    
    print("Signal visualization saved to assets/signal_examples.png")

# 直接调用生成数据
if __name__ == "__main__":
    generate_data()