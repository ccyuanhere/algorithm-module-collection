import numpy as np
import os
from scipy import signal as sig

def generate_noise(length, snr_db=None):
    """
    生成高斯白噪声
    
    Args:
        length: 噪声长度
        snr_db: 信噪比(dB)，如果为None，则生成纯噪声
    
    Returns:
        生成的噪声信号
    """
    noise = np.random.normal(0, 1, length)
    
    if snr_db is not None:
        # 将SNR从dB转换为线性单位
        snr_linear = 10 ** (snr_db / 10)
        
        # 调整噪声功率以达到所需的SNR
        noise_power = np.mean(noise ** 2)
        signal_power = noise_power * snr_linear
        
        # 生成具有所需功率的信号
        signal = np.random.normal(0, np.sqrt(signal_power), length)
        
        # 将信号和噪声混合
        mixed_signal = signal + noise
        
        return mixed_signal
    else:
        return noise

def generate_modulated_signal(length, fc, fs, modulation_type='psk', snr_db=None):
    """
    生成调制信号
    
    Args:
        length: 信号长度
        fc: 载波频率
        fs: 采样频率
        modulation_type: 调制类型，'psk', 'qam', 'fsk'
        snr_db: 信噪比(dB)
    
    Returns:
        生成的调制信号
    """
    # 生成随机比特
    bits = np.random.randint(0, 2, length)
    
    # 生成载波
    t = np.arange(length) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    
    if modulation_type == 'psk':
        # 相位键控调制
        signal = (2 * bits - 1) * carrier
    elif modulation_type == 'qam':
        # 正交幅度调制
        i_signal = (2 * bits[::2] - 1) * carrier[::2]
        q_signal = (2 * bits[1::2] - 1) * np.sin(2 * np.pi * fc * t[1::2])
        
        # 填充到原始长度
        signal = np.zeros(length, dtype=complex)
        signal[::2] = i_signal
        signal[1::2] = 1j * q_signal
    elif modulation_type == 'fsk':
        # 频率键控调制
        f0 = fc - 0.1 * fc
        f1 = fc + 0.1 * fc
        
        signal = np.zeros(length)
        for i in range(length):
            if bits[i] == 0:
                signal[i] = np.cos(2 * np.pi * f0 * t[i])
            else:
                signal[i] = np.cos(2 * np.pi * f1 * t[i])
    else:
        raise ValueError(f"不支持的调制类型: {modulation_type}")
    
    # 添加噪声
    if snr_db is not None:
        noise = generate_noise(length)
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)
        
        # 调整噪声功率以达到所需的SNR
        scaling_factor = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        noise = noise * scaling_factor
        
        # 将信号和噪声混合
        signal = signal + noise
    
    return signal

def main():
    """
    主函数：生成示例数据
    """
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"正在生成示例数据到 {data_dir}...")
    
    # 配置参数
    n_samples = 1000  # 样本数
    signal_length = 1024  # 信号长度
    fs = 1000  # 采样频率
    fc = 100  # 载波频率
    
    # 生成信号
    signals = []
    labels = []
    
    # 生成有信号的样本
    for _ in range(n_samples // 2):
        # 随机选择调制类型
        modulation_type = np.random.choice(['psk', 'qam', 'fsk'])
        
        # 随机选择信噪比
        snr_db = np.random.uniform(0, 10)
        
        # 生成调制信号
        signal = generate_modulated_signal(signal_length, fc, fs, modulation_type, snr_db)
        
        # 只取实部作为输入信号
        signals.append(np.real(signal))
        labels.append(1)  # 有信号的标签为1
    
    # 生成无信号的样本（纯噪声）
    for _ in range(n_samples // 2):
        # 生成噪声
        noise = generate_noise(signal_length)
        
        signals.append(noise)
        labels.append(0)  # 无信号的标签为0
    
    # 转换为numpy数组
    signals = np.array(signals)
    labels = np.array(labels)
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    signals = signals[indices]
    labels = labels[indices]
    
    # 保存数据
    np.save(os.path.join(data_dir, 'example_input.npy'), signals)
    np.save(os.path.join(data_dir, 'example_labels.npy'), labels)
    
    # 为了演示，我们使用默认阈值检测生成一个输出
    from model import compute_cyclic_spectrum, extract_features
    
    alpha = 0.2  # 循环频率
    nfft = 1024  # FFT点数
    threshold = 1e6  # 检测阈值
    
    detections = []
    for s in signals:
        # 计算循环谱密度
        csd = compute_cyclic_spectrum(s, fs, alpha, nfft)
        
        # 计算能量
        energy = np.sum(np.abs(csd)**2)
        
        # 使用阈值检测
        detection = 1 if energy > threshold else 0
        detections.append(detection)
    
    detections = np.array(detections)
    np.save(os.path.join(data_dir, 'example_output.npy'), detections)
    
    print(f"数据生成完成！共生成 {n_samples} 个样本。")
    
    # 计算并打印检测概率和虚警概率
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(labels, detections).ravel()
    detection_prob = tp / (tp + fn)
    false_alarm_prob = fp / (fp + tn)
    
    #print(f"初始检测概率: {detection_prob:.4f}")
    #print(f"初始虚警概率: {false_alarm_prob:.4f}")
    
    # 如果检测概率和虚警概率不符合要求，可以调整阈值
    if detection_prob < 0.9 or false_alarm_prob > 0.1:
        # 二分查找合适的阈值
        low = 1e5
        high = 1e7
        target_detection = 0.95
        target_false_alarm = 0.05
        
        for _ in range(10):  # 最多10次迭代
            mid = (low + high) / 2
            threshold = mid
            
            detections = []
            for s in signals:
                csd = compute_cyclic_spectrum(s, fs, alpha, nfft)
                energy = np.sum(np.abs(csd)**2)
                detection = 1 if energy > threshold else 0
                detections.append(detection)
            
            detections = np.array(detections)
            tn, fp, fn, tp = confusion_matrix(labels, detections).ravel()
            detection_prob = tp / (tp + fn)
            false_alarm_prob = fp / (fp + tn)
            
            if detection_prob < target_detection:
                high = mid
            else:
                low = mid
            
            if false_alarm_prob > target_false_alarm:
                low = mid
            else:
                high = mid
        
        # 使用最终找到的阈值重新生成输出
        detections = []
        for s in signals:
            csd = compute_cyclic_spectrum(s, fs, alpha, nfft)
            energy = np.sum(np.abs(csd)**2)
            detection = 1 if energy > threshold else 0
            detections.append(detection)
        
        detections = np.array(detections)
        np.save(os.path.join(data_dir, 'example_output.npy'), detections)
        
        tn, fp, fn, tp = confusion_matrix(labels, detections).ravel()
        detection_prob = tp / (tp + fn)
        false_alarm_prob = fp / (fp + tn)
        
        #print(f"调整后检测概率: {detection_prob:.4f}")
        #print(f"调整后虚警概率: {false_alarm_prob:.4f}")

if __name__ == "__main__":
    main()
