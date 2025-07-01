import os
import json
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
            "result": Any,  # 主结果（必需）
            "metrics": dict,  # 可选评估值（可为空字典）
            "log": str,  # 日志信息
            "success": bool  # 是否成功
        }
    """
    try:
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # 使用默认配置
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # 获取输入数据
        signal = input_data.get("signal")
        labels = input_data.get("labels")
        mode = input_data.get("mode", "predict")
        
        # 处理信号
        result = process(config, signal, labels, mode)
        
        # 准备返回结果
        output = {
            "result": result.get("result", None),
            "metrics": result.get("metrics", {}),
            "log": result.get("log", "处理完成"),
            "success": True
        }
        
        return output
    
    except Exception as e:
        return {
            "result": None,
            "metrics": {},
            "log": f"错误: {str(e)}",
            "success": False
        }

if __name__ == "__main__":
    # 加载示例数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    input_path = os.path.join(data_dir, 'example_input.npy')
    labels_path = os.path.join(data_dir, 'example_labels.npy')
    
    # 确保数据文件存在
    if not os.path.exists(input_path) or not os.path.exists(labels_path):
        print("示例数据不存在，正在生成...")
        make_script = os.path.join(os.path.dirname(__file__), 'make.py')
        os.system(f"python {make_script}")
    
    # 加载数据
    signal = np.load(input_path)
    labels = np.load(labels_path)
    
    # 准备输入数据
    input_data = {
        "signal": signal,
        "labels": labels,
        "mode": "evaluate"  # 评估模式
    }
    
    # 运行算法
    result = run(input_data)
    
    # 打印结果
    print("\n===== 循环平稳检测算法结果 =====")
    print(f"状态: {'成功' if result['success'] else '失败'}")
    print(f"日志: {result['log']}")
    
    if result['success'] and result['metrics']:
        print("\n性能指标:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
