import numpy as np
import os
from scipy import signal as sig
import json

def generate_example_data(num_samples=1000, signal_length=1000, snr_db=3.0, sample_rate=1000):
    """
    生成示例数据
    
    Args:
        num_samples: 样本数量
        signal_length: 信号长度
        snr_db: 信噪比(dB)
        sample_rate: 采样率(Hz)
    """
    # 创建数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成参考信号模板
    t = np.arange(0, 0.1, 1/sample_rate)  # 0.1秒的模板
    freq = sample_rate / 10
    template = np.sin(2 * np.pi * freq * t)
    template = template / np.sqrt(np.sum(template**2))
    
    # 初始化数据数组
    input_data = np.zeros((num_samples, signal_length))
    labels = np.zeros(num_samples, dtype=int)
    
    # 设置虚警概率目标 (约5%)
    false_alarm_target = 0.05
    
    # 生成信号样本
    for i in range(num_samples):
        # 随机决定是否包含信号
        has_signal = np.random.rand() < 0.5
        labels[i] = 1 if has_signal else 0
        
        # 生成噪声
        noise = np.random.randn(signal_length)
        
        if has_signal:
            # 生成信号
            signal = np.zeros(signal_length)
            # 在随机位置插入信号
            start_idx = np.random.randint(0, signal_length - len(template))
            signal[start_idx:start_idx+len(template)] = template
            
            # 调整信号幅度以达到目标SNR
            signal_power = np.sum(signal**2) / len(signal)
            noise_power = np.sum(noise**2) / len(noise)
            current_snr = 10 * np.log10(signal_power / noise_power)
            scale_factor = 10**((snr_db - current_snr) / 20)
            
            # 添加信号和噪声
            input_data[i] = scale_factor * signal + noise
        else:
            # 对于纯噪声样本，添加少量"类信号噪声"
            if np.random.rand() < false_alarm_target:
                # 生成一个与真实信号相似但不完全相同的"类信号噪声"
                fake_signal = np.zeros(signal_length)
                # 在随机位置插入一个略有变化的模板
                start_idx = np.random.randint(0, signal_length - len(template))
                
                # 创建一个与真实模板相似但略有变化的假模板
                fake_template = template * (1 + 0.1 * np.random.randn(len(template)))
                # 随机调整频率
                freq_shift = 0.05 * np.random.randn()
                fake_template = fake_template * np.exp(1j * 2 * np.pi * freq_shift * np.arange(len(template)))
                fake_template = np.real(fake_template)
                
                # 归一化假模板
                fake_template = fake_template / np.sqrt(np.sum(fake_template**2))
                
                # 插入假信号
                fake_signal[start_idx:start_idx+len(fake_template)] = fake_template
                
                # 调整假信号幅度，使其略低于真实信号
                fake_scale_factor = scale_factor * (0.7 + 0.3 * np.random.rand())
                
                # 添加假信号和噪声
                input_data[i] = fake_scale_factor * fake_signal + noise
            else:
                # 只有噪声
                input_data[i] = noise
    
    # 保存数据
    np.save(os.path.join(data_dir, 'example_input.npy'), input_data)
    np.save(os.path.join(data_dir, 'example_labels.npy'), labels)
    
    # 生成示例输出（使用当前配置运行算法）
    from model import process
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)
    
    results = []
    for i in range(num_samples):
        result = process(config, input_data[i], labels[i:i+1])
        results.append(result["detection_results"][0])
    
    np.save(os.path.join(data_dir, 'example_output.npy'), np.array(results))
    
    print(f"示例数据已生成并保存到 {data_dir}")
    print(f"样本数量: {num_samples}")
    print(f"信号长度: {signal_length}")
    print(f"信噪比: {snr_db} dB")
    print(f"包含信号的样本比例: {np.mean(labels):.2f}")
    print(f"类信号噪声样本比例: {false_alarm_target:.2f}")

if __name__ == "__main__":
    # 从配置文件加载参数
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    snr = config.get('snr', 3.0)
    sample_rate = config.get('sample_rate', 1000)
    num_samples = config.get('num_samples', 1000)
    signal_length = config.get('signal_length', 1000)
    
    # 生成示例数据
    generate_example_data(
        snr_db=snr, 
        sample_rate=sample_rate,
        num_samples=num_samples,
        signal_length=signal_length
    )
