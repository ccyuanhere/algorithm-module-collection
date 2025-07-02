import os
import json
import numpy as np
from typing import Dict, Any, Optional
from model import process

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Min-Max归一化预处理算法的标准运行接口
    
    Args:
        input_data (dict): 输入数据字典，包含以下字段：
            - signal: np.ndarray - 输入信号数据
            - normalization_range: tuple - 归一化范围，默认为(0, 1)
            - axis: int/tuple - 归一化轴，默认为None（全局归一化）
            - mode: str - 运行模式：
                * "transform": 变换模式 - 对输入信号进行归一化
                * "fit_transform": 拟合变换模式 - 计算参数并归一化
                * "inverse_transform": 逆变换模式 - 反归一化
        config_path (str, optional): 配置文件路径
        
    Returns:
        dict: 统一格式的返回结果
        {
            "result": np.ndarray,    # 归一化后的信号
            "metrics": dict,         # 归一化统计信息
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
                "default_range": [0, 1],
                "clip_outliers": False,
                "preserve_zero": False
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
        if mode not in ['transform', 'fit_transform', 'inverse_transform']:
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
        main_result = result['normalized_signal']
        metrics = result.get('statistics', {})
        
        log_info = []
        log_info.append(f"Min-Max归一化 - {mode}模式")
        log_info.append(f"输入信号形状: {signal.shape}")
        log_info.append(f"归一化范围: {result.get('normalization_range', [0, 1])}")
        
        if 'statistics' in result:
            stats = result['statistics']
            log_info.append(f"原始数据范围: [{stats.get('original_min', 0):.4f}, {stats.get('original_max', 1):.4f}]")
            log_info.append(f"归一化后范围: [{stats.get('normalized_min', 0):.4f}, {stats.get('normalized_max', 1):.4f}]")
        
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
    print("=== Min-Max归一化预处理测试 ===")
    
    # 加载测试数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        print(f"数据加载成功: {test_signal.shape}")
        print(f"原始数据范围: [{test_signal.min():.4f}, {test_signal.max():.4f}]")
    except FileNotFoundError:
        print("未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 测试不同归一化范围
    ranges = [(0, 1), (-1, 1), (0, 10)]
    
    for norm_range in ranges:
        print(f"\n--- 归一化到 {norm_range} ---")
        
        input_data = {
            "signal": test_signal,
            "normalization_range": norm_range,
            "mode": "fit_transform"
        }
        
        result = run(input_data)
        
        if result['success']:
            normalized = result['result']
            print(f"归一化后范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
            
            if result['metrics']:
                metrics = result['metrics']
                print(f"缩放因子: {metrics.get('scale_factor', 0):.6f}")
                print(f"偏移量: {metrics.get('offset', 0):.6f}")
        else:
            print(f"测试失败: {result['log']}")
    
    # 测试逆变换
    print(f"\n--- 逆变换测试 ---")
    
    # 先归一化
    input_data = {
        "signal": test_signal,
        "normalization_range": (0, 1),
        "mode": "fit_transform"
    }
    result1 = run(input_data)
    
    if result1['success']:
        # 再逆变换
        input_data = {
            "signal": result1['result'],
            "original_min": result1['metrics']['original_min'],
            "original_max": result1['metrics']['original_max'],
            "mode": "inverse_transform"
        }
        result2 = run(input_data)
        
        if result2['success']:
            recovered = result2['result']
            diff = np.mean(np.abs(test_signal - recovered))
            print(f"逆变换误差: {diff:.8f}")
        else:
            print(f"逆变换失败: {result2['log']}")

if __name__ == "__main__":
    test()
