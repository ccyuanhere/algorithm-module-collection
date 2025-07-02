import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

def min_max_normalize(signal: np.ndarray, 
                     feature_range: Tuple[float, float] = (0, 1),
                     axis: Optional[Union[int, tuple]] = None,
                     clip_outliers: bool = False,
                     outlier_percentile: float = 0.01) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Min-Max归一化函数
    
    Args:
        signal: 输入信号
        feature_range: 归一化目标范围
        axis: 归一化轴
        clip_outliers: 是否裁剪异常值
        outlier_percentile: 异常值百分位阈值
        
    Returns:
        tuple: (归一化信号, 统计信息)
    """
    
    # 处理异常值
    if clip_outliers:
        lower_percentile = outlier_percentile
        upper_percentile = 100 - outlier_percentile
        
        if axis is None:
            lower_bound = np.percentile(signal, lower_percentile)
            upper_bound = np.percentile(signal, upper_percentile)
        else:
            lower_bound = np.percentile(signal, lower_percentile, axis=axis, keepdims=True)
            upper_bound = np.percentile(signal, upper_percentile, axis=axis, keepdims=True)
        
        signal = np.clip(signal, lower_bound, upper_bound)
    
    # 计算最小值和最大值
    if axis is None:
        min_val = np.min(signal)
        max_val = np.max(signal)
    else:
        min_val = np.min(signal, axis=axis, keepdims=True)
        max_val = np.max(signal, axis=axis, keepdims=True)
    
    # 避免除零错误
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)
    
    # 归一化到[0, 1]
    normalized = (signal - min_val) / range_val
    
    # 缩放到目标范围
    target_min, target_max = feature_range
    target_range = target_max - target_min
    normalized = normalized * target_range + target_min
    
    # 统计信息
    statistics = {
        'original_min': float(np.min(min_val)),
        'original_max': float(np.max(max_val)),
        'original_range': float(np.max(max_val) - np.min(min_val)),
        'normalized_min': float(np.min(normalized)),
        'normalized_max': float(np.max(normalized)),
        'scale_factor': float(target_range / np.max(range_val)),
        'offset': float(target_min - np.min(min_val) * target_range / np.max(range_val)),
        'feature_range': feature_range,
        'axis': axis,
        'clip_outliers': clip_outliers
    }
    
    return normalized, statistics

def inverse_min_max_normalize(normalized_signal: np.ndarray,
                            original_min: float,
                            original_max: float,
                            feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Min-Max逆归一化函数
    
    Args:
        normalized_signal: 归一化后的信号
        original_min: 原始最小值
        original_max: 原始最大值
        feature_range: 归一化范围
        
    Returns:
        np.ndarray: 恢复的原始信号
    """
    target_min, target_max = feature_range
    target_range = target_max - target_min
    original_range = original_max - original_min
    
    # 先还原到[0, 1]
    normalized_01 = (normalized_signal - target_min) / target_range
    
    # 再还原到原始范围
    recovered = normalized_01 * original_range + original_min
    
    return recovered

def robust_min_max_normalize(signal: np.ndarray,
                           quantile_range: Tuple[float, float] = (0.25, 0.75),
                           feature_range: Tuple[float, float] = (0, 1),
                           axis: Optional[Union[int, tuple]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    鲁棒Min-Max归一化（基于分位数）
    
    Args:
        signal: 输入信号
        quantile_range: 分位数范围
        feature_range: 归一化目标范围
        axis: 归一化轴
        
    Returns:
        tuple: (归一化信号, 统计信息)
    """
    
    q_min, q_max = quantile_range
    
    if axis is None:
        scale_min = np.quantile(signal, q_min)
        scale_max = np.quantile(signal, q_max)
    else:
        scale_min = np.quantile(signal, q_min, axis=axis, keepdims=True)
        scale_max = np.quantile(signal, q_max, axis=axis, keepdims=True)
    
    # 避免除零
    scale_range = scale_max - scale_min
    scale_range = np.where(scale_range == 0, 1, scale_range)
    
    # 归一化
    normalized = (signal - scale_min) / scale_range
    
    # 缩放到目标范围
    target_min, target_max = feature_range
    target_range = target_max - target_min
    normalized = normalized * target_range + target_min
    
    # 统计信息
    statistics = {
        'quantile_min': float(np.min(scale_min)),
        'quantile_max': float(np.max(scale_max)),
        'quantile_range': quantile_range,
        'actual_min': float(np.min(signal)),
        'actual_max': float(np.max(signal)),
        'normalized_min': float(np.min(normalized)),
        'normalized_max': float(np.max(normalized)),
        'feature_range': feature_range,
        'method': 'robust'
    }
    
    return normalized, statistics

def process(config: dict, signal: np.ndarray, input_data: dict, mode: str = 'transform') -> Dict[str, Any]:
    """
    Min-Max归一化处理函数
    
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
        normalization_range = input_data.get('normalization_range', tuple(config.get('default_range', [0, 1])))
        axis = input_data.get('axis', None)
        clip_outliers = config.get('clip_outliers', False)
        preserve_zero = config.get('preserve_zero', False)
        method = input_data.get('method', 'standard')  # standard, robust
        
        if mode == 'transform' or mode == 'fit_transform':
            # 处理保持零值的情况
            if preserve_zero:
                # 如果需要保持零值，调整归一化范围
                if signal.min() >= 0:
                    # 全正数据，零点映射到range最小值
                    normalization_range = normalization_range
                elif signal.max() <= 0:
                    # 全负数据，零点映射到range最大值
                    normalization_range = normalization_range
                else:
                    # 有正有负，零点映射到range中心
                    center = (normalization_range[0] + normalization_range[1]) / 2
                    max_abs = max(abs(signal.min()), abs(signal.max()))
                    range_half = (normalization_range[1] - normalization_range[0]) / 2
                    scale = range_half / max_abs
                    normalized_signal = signal * scale + center
                    
                    statistics = {
                        'original_min': float(signal.min()),
                        'original_max': float(signal.max()),
                        'normalized_min': float(normalized_signal.min()),
                        'normalized_max': float(normalized_signal.max()),
                        'scale_factor': float(scale),
                        'offset': float(center),
                        'method': 'preserve_zero'
                    }
                    
                    return {
                        'normalized_signal': normalized_signal,
                        'statistics': statistics,
                        'normalization_range': normalization_range
                    }
            
            # 选择归一化方法
            if method == 'robust':
                quantile_range = input_data.get('quantile_range', (0.25, 0.75))
                normalized_signal, statistics = robust_min_max_normalize(
                    signal, quantile_range, normalization_range, axis
                )
            else:
                normalized_signal, statistics = min_max_normalize(
                    signal, normalization_range, axis, clip_outliers
                )
            
            return {
                'normalized_signal': normalized_signal,
                'statistics': statistics,
                'normalization_range': normalization_range
            }
            
        elif mode == 'inverse_transform':
            # 逆变换
            original_min = input_data.get('original_min')
            original_max = input_data.get('original_max')
            
            if original_min is None or original_max is None:
                raise ValueError("逆变换需要提供original_min和original_max参数")
            
            recovered_signal = inverse_min_max_normalize(
                signal, original_min, original_max, normalization_range
            )
            
            statistics = {
                'original_min': original_min,
                'original_max': original_max,
                'recovered_min': float(recovered_signal.min()),
                'recovered_max': float(recovered_signal.max()),
                'method': 'inverse'
            }
            
            return {
                'normalized_signal': recovered_signal,
                'statistics': statistics,
                'normalization_range': normalization_range
            }
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
            
    except Exception as e:
        return {
            'error': str(e),
            'normalized_signal': signal,
            'statistics': {}
        }
