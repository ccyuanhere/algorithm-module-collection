import os
import json
import numpy as np
from typing import Dict, Any, Optional
from model import process

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Z-score标准化预处理算法的标准运行接口
    
    Args:
        input_data (dict): 输入数据字典，包含以下字段：
            - signal: np.ndarray - 输入信号数据
            - axis: int/tuple - 标准化轴，默认为None（全局标准化）
            - ddof: int - 标准差计算的自由度修正，默认为0
            - mode: str - 运行模式：
                * "transform": 变换模式 - 使用已知参数标准化
                * "fit_transform": 拟合变换模式 - 计算统计参数并标准化
                * "inverse_transform": 逆变换模式 - 反标准化
        config_path (str, optional): 配置文件路径
        
    Returns:
        dict: 统一格式的返回结果
        {
            "result": np.ndarray,    # 标准化后的信号
            "metrics": dict,         # 标准化统计信息
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
                "ddof": 0,
                "clip_outliers": False,
                "robust_method": False
            }
        
        # 验证输入数据
        if not isinstance(input_data, dict):
            raise ValueError("input_data必须是字典类型")
        
        if 'signal' not in input_data:
            raise ValueError("input_data必须包含'signal'字段")
        
        signal = input_data['signal']
        mode = input_data.get('mode', 'fit_transform')
        
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
        main_result = result['standardized_signal']
        metrics = result.get('statistics', {})
        
        log_info = []
        log_info.append(f"Z-score标准化 - {mode}模式")
        log_info.append(f"输入信号形状: {signal.shape}")
        
        if 'statistics' in result:
            stats = result['statistics']
            log_info.append(f"原始均值: {stats.get('original_mean', 0):.4f}")
            log_info.append(f"原始标准差: {stats.get('original_std', 1):.4f}")
            log_info.append(f"标准化后均值: {stats.get('standardized_mean', 0):.4f}")
            log_info.append(f"标准化后标准差: {stats.get('standardized_std', 1):.4f}")
        
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
    print("=== Z-score标准化预处理测试 ===")
    
    # 加载测试数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        print(f"数据加载成功: {test_signal.shape}")
        print(f"原始数据统计: 均值={test_signal.mean():.4f}, 标准差={test_signal.std():.4f}")
    except FileNotFoundError:
        print("未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 测试标准化
    print(f"\n--- 标准Z-score标准化 ---")
    
    input_data = {
        "signal": test_signal,
        "mode": "fit_transform"
    }
    
    result = run(input_data)
    
    if result['success']:
        standardized = result['result']
        print(f"标准化后统计: 均值={standardized.mean():.6f}, 标准差={standardized.std():.6f}")
        
        if result['metrics']:
            metrics = result['metrics']
            print(f"原始均值: {metrics.get('original_mean', 0):.4f}")
            print(f"原始标准差: {metrics.get('original_std', 1):.4f}")
    else:
        print(f"测试失败: {result['log']}")
    
    # 测试逆变换
    print(f"\n--- 逆变换测试 ---")
    
    if result['success']:
        # 逆变换
        input_data = {
            "signal": result['result'],
            "original_mean": result['metrics']['original_mean'],
            "original_std": result['metrics']['original_std'],
            "mode": "inverse_transform"
        }
        result2 = run(input_data)
        
        if result2['success']:
            recovered = result2['result']
            diff = np.mean(np.abs(test_signal - recovered))
            print(f"逆变换误差: {diff:.8f}")
            print(f"恢复数据统计: 均值={recovered.mean():.4f}, 标准差={recovered.std():.4f}")
        else:
            print(f"逆变换失败: {result2['log']}")
    
    # 测试按轴标准化
    print(f"\n--- 按轴标准化测试 ---")
    
    # 按最后一个轴标准化
    input_data = {
        "signal": test_signal,
        "axis": -1,  # 按最后一个轴
        "mode": "fit_transform"
    }
    
    result = run(input_data)
    
    if result['success']:
        standardized = result['result']
        print(f"按轴标准化形状: {standardized.shape}")
        # 检查每个信号的最后一个轴是否标准化
        mean_per_signal = np.mean(standardized, axis=-1)
        std_per_signal = np.std(standardized, axis=-1)
        print(f"各信号均值范围: [{mean_per_signal.min():.6f}, {mean_per_signal.max():.6f}]")
        print(f"各信号标准差范围: [{std_per_signal.min():.6f}, {std_per_signal.max():.6f}]")
    else:
        print(f"按轴标准化失败: {result['log']}")

if __name__ == "__main__":
    test()
