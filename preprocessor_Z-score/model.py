import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

def zscore_standardize(signal: np.ndarray, 
                      axis: Optional[Union[int, tuple]] = None,
                      ddof: int = 0,
                      clip_outliers: bool = False,
                      outlier_std_threshold: float = 3.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Z-score标准化函数
    
    Args:
        signal: 输入信号
        axis: 标准化轴
        ddof: 标准差计算的自由度修正
        clip_outliers: 是否裁剪异常值
        outlier_std_threshold: 异常值阈值（标准差倍数）
        
    Returns:
        tuple: (标准化信号, 统计信息)
    """
    
    # 计算均值和标准差
    if axis is None:
        mean_val = np.mean(signal)
        std_val = np.std(signal, ddof=ddof)
    else:
        mean_val = np.mean(signal, axis=axis, keepdims=True)
        std_val = np.std(signal, axis=axis, keepdims=True, ddof=ddof)
    
    # 避免除零错误
    std_val = np.where(std_val == 0, 1, std_val)
    
    # Z-score标准化
    standardized = (signal - mean_val) / std_val
    
    # 处理异常值
    if clip_outliers:
        standardized = np.clip(standardized, -outlier_std_threshold, outlier_std_threshold)
    
    # 统计信息
    statistics = {
        'original_mean': float(np.mean(mean_val)),
        'original_std': float(np.mean(std_val)),
        'standardized_mean': float(np.mean(standardized)),
        'standardized_std': float(np.std(standardized)),
        'axis': axis,
        'ddof': ddof,
        'clip_outliers': clip_outliers,
        'outlier_threshold': outlier_std_threshold if clip_outliers else None
    }
    
    return standardized, statistics

def inverse_zscore_standardize(standardized_signal: np.ndarray,
                              original_mean: Union[float, np.ndarray],
                              original_std: Union[float, np.ndarray]) -> np.ndarray:
    """
    Z-score逆标准化函数
    
    Args:
        standardized_signal: 标准化后的信号
        original_mean: 原始均值
        original_std: 原始标准差
        
    Returns:
        np.ndarray: 恢复的原始信号
    """
    
    # 逆变换: X = Z * std + mean
    recovered = standardized_signal * original_std + original_mean
    
    return recovered

def robust_zscore_standardize(signal: np.ndarray,
                             axis: Optional[Union[int, tuple]] = None,
                             center_method: str = 'median',
                             scale_method: str = 'mad') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    鲁棒Z-score标准化（基于中位数和绝对中位差）
    
    Args:
        signal: 输入信号
        axis: 标准化轴
        center_method: 中心化方法 ('median', 'mean')
        scale_method: 缩放方法 ('mad', 'iqr', 'std')
        
    Returns:
        tuple: (标准化信号, 统计信息)
    """
    
    # 计算中心值
    if center_method == 'median':
        if axis is None:
            center = np.median(signal)
        else:
            center = np.median(signal, axis=axis, keepdims=True)
    else:  # mean
        if axis is None:
            center = np.mean(signal)
        else:
            center = np.mean(signal, axis=axis, keepdims=True)
    
    # 计算缩放值
    if scale_method == 'mad':
        # 绝对中位差 (Median Absolute Deviation)
        if axis is None:
            mad = np.median(np.abs(signal - center))
            scale = mad * 1.4826  # 正态分布下的一致性修正
        else:
            mad = np.median(np.abs(signal - center), axis=axis, keepdims=True)
            scale = mad * 1.4826
    elif scale_method == 'iqr':
        # 四分位距 (Interquartile Range)
        if axis is None:
            q75 = np.percentile(signal, 75)
            q25 = np.percentile(signal, 25)
            scale = (q75 - q25) / 1.349  # 正态分布下的一致性修正
        else:
            q75 = np.percentile(signal, 75, axis=axis, keepdims=True)
            q25 = np.percentile(signal, 25, axis=axis, keepdims=True)
            scale = (q75 - q25) / 1.349
    else:  # std
        if axis is None:
            scale = np.std(signal)
        else:
            scale = np.std(signal, axis=axis, keepdims=True)
    
    # 避免除零
    scale = np.where(scale == 0, 1, scale)
    
    # 鲁棒标准化
    standardized = (signal - center) / scale
    
    # 统计信息
    statistics = {
        'center_value': float(np.mean(center)),
        'scale_value': float(np.mean(scale)),
        'center_method': center_method,
        'scale_method': scale_method,
        'standardized_mean': float(np.mean(standardized)),
        'standardized_std': float(np.std(standardized)),
        'method': 'robust'
    }
    
    return standardized, statistics

def quantile_standardize(signal: np.ndarray,
                        quantile_range: Tuple[float, float] = (0.25, 0.75),
                        target_range: Tuple[float, float] = (-1, 1),
                        axis: Optional[Union[int, tuple]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    基于分位数的标准化
    
    Args:
        signal: 输入信号
        quantile_range: 分位数范围
        target_range: 目标范围
        axis: 标准化轴
        
    Returns:
        tuple: (标准化信号, 统计信息)
    """
    
    q_low, q_high = quantile_range
    target_low, target_high = target_range
    
    if axis is None:
        scale_min = np.quantile(signal, q_low)
        scale_max = np.quantile(signal, q_high)
    else:
        scale_min = np.quantile(signal, q_low, axis=axis, keepdims=True)
        scale_max = np.quantile(signal, q_high, axis=axis, keepdims=True)
    
    # 避免除零
    scale_range = scale_max - scale_min
    scale_range = np.where(scale_range == 0, 1, scale_range)
    
    # 标准化到[0, 1]
    normalized = (signal - scale_min) / scale_range
    
    # 缩放到目标范围
    target_range_val = target_high - target_low
    standardized = normalized * target_range_val + target_low
    
    # 统计信息
    statistics = {
        'quantile_low': float(np.mean(scale_min)),
        'quantile_high': float(np.mean(scale_max)),
        'quantile_range': quantile_range,
        'target_range': target_range,
        'standardized_mean': float(np.mean(standardized)),
        'standardized_std': float(np.std(standardized)),
        'method': 'quantile'
    }
    
    return standardized, statistics

def process(config: dict, signal: np.ndarray, input_data: dict, mode: str = 'fit_transform') -> Dict[str, Any]:
    """
    Z-score标准化处理函数
    
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
        axis = input_data.get('axis', None)
        ddof = input_data.get('ddof', config.get('ddof', 0))
        clip_outliers = config.get('clip_outliers', False)
        outlier_threshold = config.get('outlier_threshold', 3.0)
        method = input_data.get('method', 'standard')  # standard, robust, quantile
        
        if mode == 'transform' or mode == 'fit_transform':
            
            if method == 'robust':
                # 鲁棒标准化
                center_method = input_data.get('center_method', 'median')
                scale_method = input_data.get('scale_method', 'mad')
                standardized_signal, statistics = robust_zscore_standardize(
                    signal, axis, center_method, scale_method
                )
                
            elif method == 'quantile':
                # 分位数标准化
                quantile_range = input_data.get('quantile_range', (0.25, 0.75))
                target_range = input_data.get('target_range', (-1, 1))
                standardized_signal, statistics = quantile_standardize(
                    signal, quantile_range, target_range, axis
                )
                
            else:
                # 标准Z-score标准化
                standardized_signal, statistics = zscore_standardize(
                    signal, axis, ddof, clip_outliers, outlier_threshold
                )
            
            return {
                'standardized_signal': standardized_signal,
                'statistics': statistics
            }
            
        elif mode == 'inverse_transform':
            # 逆变换
            original_mean = input_data.get('original_mean')
            original_std = input_data.get('original_std')
            
            if original_mean is None or original_std is None:
                raise ValueError("逆变换需要提供original_mean和original_std参数")
            
            recovered_signal = inverse_zscore_standardize(
                signal, original_mean, original_std
            )
            
            statistics = {
                'original_mean': original_mean,
                'original_std': original_std,
                'recovered_mean': float(np.mean(recovered_signal)),
                'recovered_std': float(np.std(recovered_signal)),
                'method': 'inverse'
            }
            
            return {
                'standardized_signal': recovered_signal,
                'statistics': statistics
            }
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
            
    except Exception as e:
        return {
            'error': str(e),
            'standardized_signal': signal,
            'statistics': {}
        }
