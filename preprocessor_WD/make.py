import numpy as np
from pathlib import Path

def generate_signal(length=1000, batch_size=1, channels=1, noise_level=0.3):
    """
    生成带噪声的测试信号

    参数:
        length: 信号长度
        batch_size: 批量大小
        channels: 通道数
        noise_level: 噪声强度 (0-1)

    返回:
        clean_signals: 干净信号
        noisy_signals: 带噪声信号
    """
    # 创建输出目录
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    clean_signals = []
    noisy_signals = []

    for _ in range(batch_size):
        # 生成基本信号 (组合正弦波)
        t = np.linspace(0, 10, length)
        signal = np.sin(2 * np.pi * 0.5 * t)
        signal += 0.5 * np.sin(2 * np.pi * 2.3 * t + 0.1)
        signal += 0.2 * np.sin(2 * np.pi * 5.7 * t + 0.8)

        # 添加更复杂的瞬态特征
        # 尖峰1
        signal[200:205] += 1.5
        # 方波脉冲
        signal[400:450] += 0.8
        # 负尖峰
        signal[600:605] -= 1.2
        # 三角波
        signal[700:750] += np.linspace(0, 1, 50)
        signal[750:800] += np.linspace(1, 0, 50)

        # 标准化
        signal = signal - np.mean(signal)
        signal = signal / np.max(np.abs(signal))

        # 添加高斯噪声
        noise = noise_level * np.random.randn(length)
        noisy_signal = signal + noise

        clean_signals.append(signal)
        noisy_signals.append(noisy_signal)

    # 转换为numpy数组并添加通道维度
    clean_signals = np.array(clean_signals).reshape(batch_size, channels, length)
    noisy_signals = np.array(noisy_signals).reshape(batch_size, channels, length)

    # 保存文件
    np.save(data_dir / "example_input.npy", noisy_signals.astype(np.float32))
    np.save(data_dir / "example_output.npy", clean_signals.astype(np.float32))
    
    # 生成标签数据
    labels = np.ones((batch_size, 1))  # 这里简单生成全1的标签，你可以根据实际需求修改
    np.save(data_dir / "example_labels.npy", labels.astype(np.float32))

    print(f"Generated signals saved to {data_dir}/")
    print(f"Signal shape: {noisy_signals.shape}")
    print(f"Generated labels saved to {data_dir}/example_labels.npy")

if __name__ == "__main__":
    generate_signal(length=1000, batch_size=1, channels=1, noise_level=0.3)