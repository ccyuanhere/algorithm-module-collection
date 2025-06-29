import json
import os
import time
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
            "result": Any,        # 主结果（必需）
            "metrics": dict,      # 可选评估值（可为空字典）
            "log": str,          # 日志信息
            "success": bool      # 是否成功
        }
    """
    try:
        # 设置默认配置文件路径
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        # 加载配置文件
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 提取输入数据
        signal = input_data['signal']
        labels = input_data.get('labels', None)
        mode = input_data.get('mode', 'predict')
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用核心处理函数
        result = process(config, signal, labels, mode)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        avg_time_per_sample = processing_time / len(signal) if len(signal) > 0 else 0
        
        # 构建返回结果
        if mode == 'evaluate' and 'metrics' in result:
            result['metrics']['processing_time'] = processing_time
            result['metrics']['avg_time_per_sample'] = avg_time_per_sample
        else:
            result.setdefault('metrics', {})['processing_time'] = processing_time
            result['metrics']['avg_time_per_sample'] = avg_time_per_sample
        
        return {
            "result": result.get('result', None),
            "metrics": result.get('metrics', {}),
            "log": result.get('log', "Success"),
            "success": True
        }
    
    except Exception as e:
        return {
            "result": None,
            "metrics": {},
            "log": f"Error: {str(e)}",
            "success": False
        }

# 测试代码
if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs('data', exist_ok=True)
    
    # 如果数据文件不存在，则生成数据
    if not os.path.exists('data/example_input.npy'):
        print("Generating example data...")
        import make
        make.generate_data()
    
    # 加载示例数据
    signals = np.load('data/example_input.npy')
    labels = np.load('data/example_labels.npy')
    
    # 第一步：训练模型
    print("Training model...")
    train_result = run({
        "signal": signals,
        "labels": labels,
        "mode": "train"
    })
    
    if not train_result['success']:
        print(f"Training failed: {train_result['log']}")
        exit(1)
    
    print(f"Training completed. Model saved to models/wtd_model.pkl")
    
    # 第二步：评估模型
    print("\nEvaluating model...")
    eval_result = run({
        "signal": signals,
        "labels": labels,
        "mode": "evaluate"
    })
    
    if not eval_result['success']:
        print(f"Evaluation failed: {eval_result['log']}")
        exit(1)
    
    # 打印结果
    print("\nEvaluation completed:")
    print(f"Success: {eval_result['success']}")
    print(f"Log: {eval_result['log']}")
    print("Metrics:")
    for k, v in eval_result['metrics'].items():
        print(f"  {k}: {v:.4f}")
    
    # 保存输出结果
    if eval_result['result'] is not None:
        np.save('data/example_output.npy', eval_result['result'])
        print("\nOutput saved to data/example_output.npy")
    
    # 可视化特征分布
    if 'feature_distribution' in eval_result['metrics']:
        print("\nFeature distribution visualization saved to assets/feature_distribution.png")