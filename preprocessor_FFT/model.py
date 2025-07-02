import numpy as np
from typing import Dict, Any, Optional, Union
from scipy import signal as scipy_signal

def apply_window(signal: np.ndarray, window_type: str = 'hann', axis: int = -1) -> np.ndarray:
    """
    应用窗函数
    
    Args:
        signal: 输入信号
        window_type: 窗函数类型
        axis: 应用窗函数的轴
        
    Returns:
        np.ndarray: 加窗后的信号
    """
    
    signal_length = signal.shape[axis]
    
    if window_type == 'hann':
        window = np.hanning(signal_length)
    elif window_type == 'hamming':
        window = np.hamming(signal_length)
    elif window_type == 'blackman':
        window = np.blackman(signal_length)
    elif window_type == 'bartlett':
        window = np.bartlett(signal_length)
    elif window_type == 'kaiser':
        window = np.kaiser(signal_length, beta=8.6)  # beta=8.6对应约60dB衰减
    elif window_type == 'rectangular' or window_type == 'none':
        window = np.ones(signal_length)
    else:
        raise ValueError(f"不支持的窗函数类型: {window_type}")
    
    # 扩展窗函数到信号的维度
    window_shape = [1] * signal.ndim
    window_shape[axis] = signal_length
    window = window.reshape(window_shape)
    
    return signal * window

def compute_fft(signal: np.ndarray, 
                fft_type: str = 'fft',
                zero_padding: Optional[int] = None,
                axis: int = -1,
                normalize: bool = False) -> np.ndarray:
    """
    计算FFT变换
    
    Args:
        signal: 输入信号
        fft_type: FFT类型
        zero_padding: 零填充长度
        axis: FFT轴
        normalize: 是否归一化
        
    Returns:
        np.ndarray: FFT结果
    """
    
    # 零填充
    if zero_padding is not None and zero_padding > signal.shape[axis]:
        pad_width = [(0, 0)] * signal.ndim
        pad_width[axis] = (0, zero_padding - signal.shape[axis])
        signal = np.pad(signal, pad_width, mode='constant')
    
    # 执行FFT
    if fft_type == 'fft':
        result = np.fft.fft(signal, axis=axis)
        if normalize:
            result = result / signal.shape[axis]
    elif fft_type == 'rfft':
        result = np.fft.rfft(signal, axis=axis)
        if normalize:
            result = result / signal.shape[axis]
    elif fft_type == 'ifft':
        result = np.fft.ifft(signal, axis=axis)
    elif fft_type == 'irfft':
        result = np.fft.irfft(signal, axis=axis)
    else:
        raise ValueError(f"不支持的FFT类型: {fft_type}")
    
    return result

def format_fft_output(fft_result: np.ndarray, 
                     return_format: str = 'complex') -> np.ndarray:
    """
    格式化FFT输出
    
    Args:
        fft_result: FFT结果（复数）
        return_format: 返回格式
        
    Returns:
        np.ndarray: 格式化后的结果
    """
    
    if return_format == 'complex':
        return fft_result
    elif return_format == 'magnitude':
        return np.abs(fft_result)
    elif return_format == 'phase':
        return np.angle(fft_result)
    elif return_format == 'magnitude_phase':
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        return np.stack([magnitude, phase], axis=-1)
    elif return_format == 'power':
        return np.abs(fft_result) ** 2
    elif return_format == 'power_db':
        power = np.abs(fft_result) ** 2
        return 10 * np.log10(power + 1e-12)  # 避免log(0)
    elif return_format == 'real_imag':
        return np.stack([fft_result.real, fft_result.imag], axis=-1)
    else:
        raise ValueError(f"不支持的返回格式: {return_format}")

def compute_frequency_axis(signal_length: int, 
                          sampling_rate: float = 1.0,
                          fft_type: str = 'fft') -> np.ndarray:
    """
    计算频率轴
    
    Args:
        signal_length: 信号长度
        sampling_rate: 采样率
        fft_type: FFT类型
        
    Returns:
        np.ndarray: 频率轴
    """
    
    if fft_type == 'fft':
        freqs = np.fft.fftfreq(signal_length, 1/sampling_rate)
    elif fft_type == 'rfft':
        freqs = np.fft.rfftfreq(signal_length, 1/sampling_rate)
    else:
        freqs = np.fft.fftfreq(signal_length, 1/sampling_rate)
    
    return freqs

def analyze_spectrum(fft_result: np.ndarray,
                    frequencies: np.ndarray,
                    signal_length: int) -> Dict[str, Any]:
    """
    分析频谱特征
    
    Args:
        fft_result: FFT结果
        frequencies: 频率轴
        signal_length: 原始信号长度
        
    Returns:
        dict: 频谱分析结果
    """
    
    # 计算功率谱
    power_spectrum = np.abs(fft_result) ** 2
    
    # 找到主频率
    max_power_idx = np.argmax(power_spectrum)
    dominant_frequency = frequencies[max_power_idx]
    
    # 计算总功率
    total_power = np.sum(power_spectrum)
    
    # 计算频率分辨率
    frequency_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
    
    # 计算有效频带
    threshold = 0.1 * np.max(power_spectrum)  # 10%阈值
    significant_indices = np.where(power_spectrum > threshold)[0]
    
    if len(significant_indices) > 0:
        frequency_range = (frequencies[significant_indices[0]], 
                          frequencies[significant_indices[-1]])
    else:
        frequency_range = (frequencies[0], frequencies[-1])
    
    # 计算谱重心（质心频率）
    if total_power > 0:
        spectral_centroid = np.sum(frequencies * power_spectrum) / total_power
    else:
        spectral_centroid = 0
    
    # 计算谱带宽
    if total_power > 0:
        spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * power_spectrum) / total_power)
    else:
        spectral_bandwidth = 0
    
    # 计算谱平坦度（谱的均匀程度）
    geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-12)))
    arithmetic_mean = np.mean(power_spectrum)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-12)
    
    return {
        'dominant_frequency': float(dominant_frequency),
        'dominant_frequency_power': float(power_spectrum[max_power_idx]),
        'total_power': float(total_power),
        'frequency_resolution': float(frequency_resolution),
        'frequency_range': frequency_range,
        'spectral_centroid': float(spectral_centroid),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_flatness': float(spectral_flatness),
        'num_frequency_bins': len(frequencies),
        'signal_length': signal_length
    }

def process(config: dict, signal: np.ndarray, input_data: dict, mode: str = 'transform') -> Dict[str, Any]:
    """
    FFT变换处理函数
    
    Args:
        config: 配置参数
        signal: 输入信号
        input_data: 输入数据字典
        mode: 运行模式
        
    Returns:
        dict: 处理结果
    """
    
    try:
        # 获取参数
        fft_type = input_data.get('fft_type', config.get('default_fft_type', 'fft'))
        return_format = input_data.get('return_format', config.get('default_return_format', 'complex'))
        window_type = input_data.get('window', 'none')
        zero_padding = input_data.get('zero_padding', None)
        sampling_rate = input_data.get('sampling_rate', 1.0)
        axis = input_data.get('axis', -1)
        normalize = config.get('normalize_output', False)
        
        original_signal = signal.copy()
        original_shape = signal.shape
        
        if mode == 'transform':
            # 前向变换模式
            
            # 应用窗函数
            if window_type != 'none' and config.get('apply_window', True):
                signal = apply_window(signal, window_type, axis)
            
            # 执行FFT
            fft_result = compute_fft(signal, fft_type, zero_padding, axis, normalize)
            
            # 格式化输出
            if return_format != 'complex':
                formatted_result = format_fft_output(fft_result, return_format)
            else:
                formatted_result = fft_result
            
            return {
                'fft_result': formatted_result,
                'fft_type': fft_type,
                'return_format': return_format,
                'original_shape': original_shape,
                'window_applied': window_type,
                'zero_padding': zero_padding
            }
            
        elif mode == 'inverse':
            # 逆变换模式
            
            # 假设输入是复数FFT结果
            if fft_type == 'ifft':
                result = np.fft.ifft(signal, axis=axis)
            elif fft_type == 'irfft':
                result = np.fft.irfft(signal, axis=axis)
            else:
                # 默认使用ifft
                result = np.fft.ifft(signal, axis=axis)
            
            # 如果结果是复数但原信号可能是实数，取实部
            if np.iscomplexobj(result) and not np.iscomplexobj(original_signal):
                result = result.real
            
            return {
                'fft_result': result,
                'fft_type': fft_type,
                'return_format': 'time_domain',
                'original_shape': original_shape
            }
            
        elif mode == 'analyze':
            # 分析模式
            
            # 应用窗函数
            if window_type != 'none' and config.get('apply_window', True):
                signal = apply_window(signal, window_type, axis)
            
            # 执行FFT
            fft_result = compute_fft(signal, 'fft', zero_padding, axis, normalize)
            
            # 计算频率轴
            signal_length = original_signal.shape[axis]
            frequencies = compute_frequency_axis(signal_length, sampling_rate, 'fft')
            
            # 如果是多维信号，只分析最后一个轴
            if signal.ndim > 1:
                # 取一个代表性的切片进行分析
                analysis_slice = tuple([0] * (signal.ndim - 1) + [slice(None)])
                fft_slice = fft_result[analysis_slice]
                freq_analysis = analyze_spectrum(fft_slice, frequencies, signal_length)
            else:
                freq_analysis = analyze_spectrum(fft_result, frequencies, signal_length)
            
            # 格式化输出
            formatted_result = format_fft_output(fft_result, return_format)
            
            return {
                'fft_result': formatted_result,
                'analysis': freq_analysis,
                'frequencies': frequencies,
                'fft_type': 'fft',
                'return_format': return_format,
                'sampling_rate': sampling_rate
            }
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
            
    except Exception as e:
        return {
            'error': str(e),
            'fft_result': signal,
            'analysis': {}
        }
