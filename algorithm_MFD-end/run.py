import json
import os
import numpy as np
from typing import Dict, Any, Optional
from model import process

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    标准运行接口 - 所有算法的统一入口
    
    Args:
        input_data (dict): 输入数据字典，根据算法类型包含不同字段
        config_path (str, optional): 配置文件路径
        
    Returns:
        dict: 统一格式的返回结果
        {
            "result": Any,          # 主结果（必需）
            "metrics": dict,        # 可选评估值（可为空字典）
            "log": str,             # 日志信息
            "success": bool         # 是否成功
        }
    """
    try:
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # 使用默认配置
            default_config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(default_config_path, 'r') as f:
                config = json.load(f)
        
        # 获取输入数据
        signal = input_data.get('signal', None)
        labels = input_data.get('labels', None)
        mode = input_data.get('mode', 'predict')
        
        # 检查输入
        if signal is None:
            return {
                "result": None,
                "metrics": {},
                "log": "错误: 缺少输入信号",
                "success": False
            }
        
        # 处理数据
        result = process(config, signal, labels)
        
        # 生成日志
        log = f"匹配滤波检测算法执行完成，模式: {mode}"
        
        return {
            "result": result,
            "metrics": result.get('metrics', {}),
            "log": log,
            "success": True
        }
    
    except Exception as e:
        return {
            "result": None,
            "metrics": {},
            "log": f"错误: {str(e)}",
            "success": False
        }

if __name__ == "__main__":
    # 加载示例数据
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    input_data = {
        "signal": np.load(os.path.join(data_path, 'example_input.npy')),
        "labels": np.load(os.path.join(data_path, 'example_labels.npy')),
        "mode": "evaluate"
    }
    
    # 运行算法
    result = run(input_data)
    
    # 输出结果
    print("匹配滤波检测算法执行结果:")
    print(f"成功: {result['success']}")
    print(f"日志: {result['log']}")
    
    if result['success'] and 'metrics' in result['result']:
        metrics = result['result']['metrics']
        print("\n性能指标:")
        print(f"检测概率 (Pd): {metrics['detection_probability']:.4f}")
        print(f"虚警概率 (Pf): {metrics['false_alarm_probability']:.4f}")
        print(f"信噪比 (SNR): {metrics['snr']:.2f} dB")
        print(f"平均检测时间: {metrics['avg_detection_time']:.4f} 秒")
