import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from model import process

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    FFT（快速傅里叶变换）预处理算法的标准运行接口
    
    Args:
        input_data (dict): 输入数据字典，包含以下字段：
            - signal: np.ndarray - 输入时域信号数据
            - fft_type: str - FFT类型：'fft', 'rfft', 'ifft', 'irfft'
            - return_format: str - 返回格式：'complex', 'magnitude', 'phase', 'magnitude_phase', 'power'
            - window: str - 窗函数类型（可选）
            - zero_padding: int - 零填充长度（可选）
            - mode: str - 运行模式：
                * "transform": 变换模式 - 执行FFT变换
                * "inverse": 逆变换模式 - 执行IFFT逆变换
                * "analyze": 分析模式 - 进行频谱分析
        config_path (str, optional): 配置文件路径
        
    Returns:
        dict: 统一格式的返回结果
        {
            "result": np.ndarray,    # FFT变换结果
            "metrics": dict,         # 频谱分析统计信息
            "log": str,             # 日志信息
            "success": bool         # 是否成功
        }
    """
    
    try:
        # 加载配置
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 默认配置
            config = {
                "default_fft_type": "fft",
                "default_return_format": "complex",
                "apply_window": True,
                "normalize_output": False
            }
        
        # 验证输入数据
        if not isinstance(input_data, dict):
            raise ValueError("input_data必须是字典类型")
        
        if 'signal' not in input_data:
            raise ValueError("input_data必须包含'signal'字段")
        
        signal = input_data['signal']
        mode = input_data.get('mode', 'transform')
        
        # 验证信号格式
        if not isinstance(signal, np.ndarray):
            raise ValueError("signal必须是numpy数组")
        
        # 验证模式
        if mode not in ['transform', 'inverse', 'analyze']:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 调用处理函数
        result = process(config, signal, input_data, mode)
        
        # 检查处理结果
        if 'error' in result:
            return {
                "result": np.array([]),
                "metrics": {},
                "log": f"处理错误: {result['error']}",
                "success": False
            }
        
        # 构建返回结果
        main_result = result['fft_result']
        metrics = result.get('analysis', {})
        
        log_info = []
        log_info.append(f"FFT变换 - {mode}模式")
        log_info.append(f"输入信号形状: {signal.shape}")
        
        if 'fft_type' in result:
            log_info.append(f"FFT类型: {result['fft_type']}")
        
        if 'return_format' in result:
            log_info.append(f"返回格式: {result['return_format']}")
        
        if 'analysis' in result and mode == 'analyze':
            analysis = result['analysis']
            if 'dominant_frequency' in analysis:
                log_info.append(f"主频率: {analysis['dominant_frequency']:.2f} Hz")
            if 'frequency_resolution' in analysis:
                log_info.append(f"频率分辨率: {analysis['frequency_resolution']:.4f} Hz")
        
        return {
            "result": main_result,
            "metrics": metrics,
            "log": "\n".join(log_info),
            "success": True
        }
        
    except Exception as e:
        return {
            "result": np.array([]),
            "metrics": {},
            "log": f"运行错误: {str(e)}",
            "success": False
        }

def visualize_fft_analysis(signal, fft_result, sampling_rate=1000, signal_index=0, save_dir=None):
    """
    可视化FFT分析结果
    
    Args:
        signal: 原始信号（可能是复数）
        fft_result: FFT变换结果
        sampling_rate: 采样率
        signal_index: 信号索引
        save_dir: 保存目录
    """
    
    # 确保保存目录存在
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'assets')
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建时间和频率轴
    N = len(signal)
    time_axis = np.arange(N) / sampling_rate
    freq_axis = np.fft.fftfreq(N, 1/sampling_rate)
    
    # 判断是否为复数信号
    is_complex = np.iscomplexobj(signal)
    
    # 创建图形
    if is_complex:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'FFT变换分析 - 复数信号 #{signal_index}', fontsize=16)
        
        # 时域信号 - 实部和虚部
        axes[0, 0].plot(time_axis, signal.real, 'b-', label='实部', linewidth=0.8)
        axes[0, 0].plot(time_axis, signal.imag, 'r-', label='虚部', linewidth=0.8)
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('信号幅度')
        axes[0, 0].set_title('时域信号')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 复数信号的模长
        magnitude = np.abs(signal)
        axes[0, 1].plot(time_axis, magnitude, 'g-', linewidth=0.8)
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('信号模长')
        axes[0, 1].set_title('复数信号模长')
        axes[0, 1].grid(True, alpha=0.3)
        
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'FFT变换分析 - 实数信号 #{signal_index}', fontsize=16)
        
        # 时域信号
        axes[0, 0].plot(time_axis, signal, 'b-', linewidth=0.8)
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('信号幅度')
        axes[0, 0].set_title('时域信号')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 信号统计信息
        signal_stats = {
            '均值': np.mean(signal),
            '标准差': np.std(signal),
            '最大值': np.max(signal),
            '最小值': np.min(signal),
            '峰峰值': np.max(signal) - np.min(signal)
        }
        
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in signal_stats.items()])
        axes[0, 1].text(0.1, 0.9, stats_text, transform=axes[0, 1].transAxes, 
                       fontsize=12, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[0, 1].set_title('信号统计信息')
        axes[0, 1].axis('off')
    
    # 幅度谱
    magnitude_spectrum = np.abs(fft_result)
    # 只显示正频率部分
    half_N = N // 2
    axes[1, 0].plot(freq_axis[:half_N], magnitude_spectrum[:half_N], 'r-', linewidth=0.8)
    axes[1, 0].set_xlabel('频率 (Hz)')
    axes[1, 0].set_ylabel('幅度')
    axes[1, 0].set_title('幅度谱')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 相位谱
    phase_spectrum = np.angle(fft_result)
    axes[1, 1].plot(freq_axis[:half_N], phase_spectrum[:half_N], 'g-', linewidth=0.8)
    axes[1, 1].set_xlabel('频率 (Hz)')
    axes[1, 1].set_ylabel('相位 (弧度)')
    axes[1, 1].set_title('相位谱')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'fft_analysis_signal_{signal_index}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def test():
    """测试函数，展示FFT变换效果"""
    print("=== FFT变换预处理测试 ===")
    
    # 加载测试数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        print(f"数据加载成功: {test_signal.shape}")
        print(f"信号数据类型: {test_signal.dtype}")
    except FileNotFoundError:
        print("未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 选择几个代表性信号进行测试
    test_indices = [0, 20, 40, 80, 100]  # 包含实数和复数信号
    sampling_rate = 1000.0
    
    for i, idx in enumerate(test_indices):
        if idx >= len(test_signal):
            continue
            
        signal_data = test_signal[idx]  # 形状: (2, 1024)
        
        # 构造复数信号
        if idx < 80:  # 实数信号
            signal = signal_data[0]  # 只取实部
            signal_type = "实数"
        else:  # 复数信号
            signal = signal_data[0] + 1j * signal_data[1]
            signal_type = "复数"
        
        print(f"\n--- 测试信号 #{idx} ({signal_type}) ---")
        print(f"信号形状: {signal.shape}")
        print(f"数据类型: {signal.dtype}")
        print(f"数值范围: [{np.min(np.abs(signal)):.4f}, {np.max(np.abs(signal)):.4f}]")
        
        # 执行FFT变换
        input_data = {
            "signal": signal,
            "fft_type": "fft",
            "return_format": "complex",
            "mode": "analyze",
            "sampling_rate": sampling_rate
        }
        
        result = run(input_data)
        
        if result['success']:
            fft_result = result['result']
            metrics = result['metrics']
            
            print(f"FFT变换成功:")
            print(f"  输出形状: {fft_result.shape}")
            print(f"  主频率: {metrics.get('dominant_frequency', 'N/A')} Hz")
            print(f"  总功率: {metrics.get('total_power', 'N/A'):.4f}")
            print(f"  频率分辨率: {metrics.get('frequency_resolution', 'N/A')} Hz")
            
            # 生成可视化
            filepath = visualize_fft_analysis(signal, fft_result, sampling_rate, idx)
            print(f"  可视化保存到: {filepath}")
            
        else:
            print(f"FFT变换失败: {result['log']}")
    
    # 展示不同返回格式的效果
    print(f"\n--- 不同返回格式测试 ---")
    
    test_signal_single = test_signal[0][0]  # 取第一个实数信号
    return_formats = ['complex', 'magnitude', 'phase', 'power']
    
    for return_format in return_formats:
        input_data = {
            "signal": test_signal_single,
            "return_format": return_format,
            "mode": "transform"
        }
        
        result = run(input_data)
        
        if result['success']:
            output = result['result']
            print(f"  {return_format}: 形状 {output.shape}, 数据类型 {output.dtype}")
            if np.iscomplexobj(output):
                print(f"    数值范围: 实部[{output.real.min():.4f}, {output.real.max():.4f}], 虚部[{output.imag.min():.4f}, {output.imag.max():.4f}]")
            else:
                print(f"    数值范围: [{output.min():.4f}, {output.max():.4f}]")
        else:
            print(f"  {return_format}: 失败")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test()
