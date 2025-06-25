import json
import os
from typing import Dict, Any, Optional
import numpy as np
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
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # 使用默认配置
            default_config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(default_config_path, 'r') as f:
                config = json.load(f)
        
        # 提取输入数据
        signal = input_data.get('signal', None)
        labels = input_data.get('labels', None)
        mode = input_data.get('mode', 'predict')
        
        # 验证输入
        if signal is None:
            raise ValueError("输入数据中缺少'signal'字段")
        
        # 执行算法
        result = process(config, signal, labels, mode)
        
        # 构建返回结果
        output = {
            "result": result.get("detection_result", None),
            "metrics": result.get("metrics", {}),
            "log": result.get("log", "能量检测算法执行成功"),
            "success": True
        }
        
        return output
    
    except Exception as e:
        # 异常处理
        return {
            "result": None,
            "metrics": {},
            "log": f"能量检测算法执行失败: {str(e)}",
            "success": False
        }

if __name__ == "__main__":
    # 示例：如何使用run函数
    # 注意：实际运行时需要确保data目录存在并包含正确的文件
    
    try:
        # 加载示例数据
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        input_file = os.path.join(data_dir, 'example_input.npy')
        labels_file = os.path.join(data_dir, 'example_labels.npy')
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"示例输入文件不存在: {input_file}")
        
        signal = np.load(input_file)
        labels = np.load(labels_file) if os.path.exists(labels_file) else None
        
        # 准备输入数据
        input_data = {
            "signal": signal,
            "labels": labels,
            "mode": "evaluate"  # 可选: "train", "predict", "evaluate"
        }
        
        # 运行算法
        result = run(input_data)
        
        # 输出结果
        print("\n===== 能量检测算法运行结果 =====")
        print(f"是否成功: {result['success']}")
        print(f"日志: {result['log']}")
        
        if result['success'] and result['result'] is not None:
            print(f"\n检测结果 (前5个样本):")
            print(result['result'][:5])
            
            if result['metrics']:
                print("\n评估指标:")
                for metric, value in result['metrics'].items():
                    print(f"  {metric}: {value:.4f}")
        else:
            print("未获得有效的检测结果")
            
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()
