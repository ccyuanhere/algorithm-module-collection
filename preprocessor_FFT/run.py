import os
import json
import numpy as np
from typing import Dict, Any, Optional
from model import process

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

def test():
    """简单的测试函数"""
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
    
    # 测试不同FFT类型
    fft_types = ['fft', 'rfft']
    return_formats = ['complex', 'magnitude', 'phase', 'power']
    
    for fft_type in fft_types:
        print(f"\n--- {fft_type.upper()}变换测试 ---")
        
        for return_format in return_formats:
            if fft_type == 'rfft' and return_format == 'complex':
                continue  # RFFT通常不返回复数格式用于演示
            
            print(f"\n{return_format}格式:")
            
            input_data = {
                "signal": test_signal[0],  # 测试第一个信号
                "fft_type": fft_type,
                "return_format": return_format,
                "mode": "transform"
            }
            
            result = run(input_data)
            
            if result['success']:
                fft_result = result['result']
                print(f"  输出形状: {fft_result.shape}")
                print(f"  数据类型: {fft_result.dtype}")
                if np.iscomplexobj(fft_result):
                    print(f"  数值范围: 实部[{fft_result.real.min():.2f}, {fft_result.real.max():.2f}], 虚部[{fft_result.imag.min():.2f}, {fft_result.imag.max():.2f}]")
                else:
                    print(f"  数值范围: [{fft_result.min():.2f}, {fft_result.max():.2f}]")
            else:
                print(f"  测试失败: {result['log']}")
    
    # 测试频谱分析模式
    print(f"\n--- 频谱分析模式测试 ---")
    
    input_data = {
        "signal": test_signal[0],
        "sampling_rate": 1000,  # 1kHz采样率
        "mode": "analyze"
    }
    
    result = run(input_data)
    
    if result['success']:
        analysis = result['metrics']
        print(f"主频率: {analysis.get('dominant_frequency', 'N/A')} Hz")
        print(f"频率分辨率: {analysis.get('frequency_resolution', 'N/A')} Hz")
        print(f"信号功率: {analysis.get('total_power', 'N/A')}")
        print(f"频带范围: {analysis.get('frequency_range', 'N/A')} Hz")
    else:
        print(f"分析失败: {result['log']}")
    
    # 测试窗函数
    print(f"\n--- 窗函数测试 ---")
    
    windows = ['hann', 'hamming', 'blackman', 'kaiser']
    
    for window in windows:
        input_data = {
            "signal": test_signal[0],
            "window": window,
            "return_format": "magnitude",
            "mode": "transform"
        }
        
        result = run(input_data)
        
        if result['success']:
            print(f"  {window}窗: 成功，输出形状 {result['result'].shape}")
        else:
            print(f"  {window}窗: 失败")

if __name__ == "__main__":
    test()
