import numpy as np
import os

def generate_example_data():
    """生成更真实的示例数据（纯NumPy实现）"""
    os.makedirs("data", exist_ok=True)

    # 生成基础信号 (10个样本，1个通道，2000个点)
    time = np.linspace(0, 2, 2000)
    input_signals = []

    for i in range(10):
        # 基础波形 (多种频率的正弦波+噪声)
        base = 0.7 * np.sin(2 * np.pi * 8 * time)
        base += 0.4 * np.sin(2 * np.pi * 25 * time)
        base += 0.3 * np.sin(2 * np.pi * 60 * time)
        base += 0.2 * np.random.normal(size=len(time))

        # 添加随机峰值 - 增加峰值数量和幅度
        num_peaks = np.random.randint(15, 30)
        peak_positions = np.random.choice(len(time), num_peaks, replace=False)
        peak_amplitudes = np.random.uniform(2.0, 4.0, num_peaks)

        for pos, amp in zip(peak_positions, peak_amplitudes):
            # 创建更宽的高斯脉冲
            x = np.linspace(-4.0, 4.0, 50)
            pulse = amp * np.exp(-x**2/(2*2.0**2))  # 更宽的高斯

            start = max(0, pos - 25)
            end = min(len(time), pos + 25)
            pulse_len = end - start

            if pulse_len < 50:
                base[start:end] += pulse[:pulse_len]
            else:
                base[start:end] += pulse[:pulse_len]

        input_signals.append(base)

    input_signal = np.array(input_signals).reshape(10, 1, 2000)

    # 生成与信号特征相关的标签
    peak_counts = np.array([np.sum(np.abs(s) > 1.8) for s in input_signal[:, 0]])
    labels = (peak_counts > np.median(peak_counts)).astype(int).reshape(-1, 1)

    # 生成输出信号 (使用软裁剪算法)
    output_signal = np.zeros_like(input_signal)
    threshold = 0.6  # 使用与config.json相同的默认阈值
    alpha = 2.0      # 使用与config.json相同的默认平滑因子

    for i in range(input_signal.shape[0]):
        for j in range(input_signal.shape[1]):
            # 应用软裁剪
            output_signal[i, j] = threshold * np.tanh(input_signal[i, j] * alpha / threshold)

    # 保存所有文件
    np.save("data/example_input.npy", input_signal.astype(np.float32))
    np.save("data/example_labels.npy", labels.astype(np.int32))
    np.save("data/example_output.npy", output_signal.astype(np.float32))

    print("Enhanced example data generated in data/ directory")
    print(f"  - Input data shape: {input_signal.shape}")
    print(f"  - Output data shape: {output_signal.shape}")
    print(f"  - Labels shape: {labels.shape}")

if __name__ == "__main__":
    generate_example_data()