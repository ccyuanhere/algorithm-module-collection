import numpy as np
import json
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
            "result": Any,        # 主结果（必需）
            "metrics": dict,      # 可选评估值（可为空字典）
            "log": str,          # 日志信息
            "success": bool      # 是否成功
        }
    """
    try:
        # 加载默认配置
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # 如果提供了外部配置文件，则覆盖默认配置
        if config_path:
            with open(config_path, 'r') as f:
                external_config = json.load(f)
                config.update(external_config)
        
        # 获取输入数据
        signal = input_data.get("signal", None)
        labels = input_data.get("labels", None)
        mode = input_data.get("mode", "evaluate")
        
        if signal is None:
            return {
                "result": None,
                "metrics": {},
                "log": "输入信号缺失",
                "success": False
            }
        
        # 调用核心处理函数
        results = process(config, signal, labels)
        
        # 返回结果
        return {
            "result": results.get("result", None),
            "metrics": results.get("metrics", {}),
            "log": "算法执行成功",
            "success": True
        }
    
    except Exception as e:
        return {
            "result": None,
            "metrics": {},
            "log": f"算法执行错误: {str(e)}",
            "success": False
        }

# 示例测试代码
if __name__ == "__main__":
    # 加载示例数据
    input_signal = np.load("data/example_input.npy")
    labels = np.load("data/example_labels.npy")
    
    # 准备输入数据
    input_data = {
        "signal": input_signal,
        "labels": labels,
        "mode": "evaluate"
    }
    
    # 执行算法
    output = run(input_data)
    
    # 打印结果
    print("执行结果:", output["success"])
    print("日志信息:", output["log"])
    
    if output["success"]:
        print("\n性能指标:")
        for key, value in output["metrics"].items():
            print(f"{key}: {value}")