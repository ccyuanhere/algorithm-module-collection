import os
import json
import numpy as np
from typing import Dict, Any, Optional
from model import process

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    匹配滤波检测算法的标准运行接口
    
    Args:
        input_data (dict): 输入数据字典，包含以下字段：
            - signal: np.ndarray - 输入I/Q信号 (N, signal_length, 2) 或 (signal_length, 2)
            - labels: np.ndarray - 标签数据 (N,) - 0表示无信号，1表示有信号 (evaluate/train模式需要)
            - mode: str - 运行模式：
                * "predict": 预测模式 - 对输入信号进行检测，输出检测结果
                * "evaluate": 评估模式 - 需要labels，计算性能指标
                * "train": 训练模式 - 对传统算法，用于参数校准
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
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 默认配置
            config = {
                "threshold": 0.3,
                "template_type": "bpsk",
                "samples_per_bit": 8
            }
        
        # 验证输入数据
        if not isinstance(input_data, dict):
            raise ValueError("input_data必须是字典类型")
        
        if 'signal' not in input_data:
            raise ValueError("input_data必须包含'signal'字段")
        
        if 'mode' not in input_data:
            raise ValueError("input_data必须包含'mode'字段")
        
        signal = input_data['signal']
        labels = input_data.get('labels', None)
        mode = input_data['mode']
        
        # 验证信号格式
        if not isinstance(signal, np.ndarray):
            raise ValueError("signal必须是numpy数组")
        
        if signal.ndim not in [2, 3]:
            raise ValueError("signal必须是2D (signal_length, 2) 或 3D (N, signal_length, 2) 数组")
        
        if signal.shape[-1] != 2:
            raise ValueError("signal的最后一维必须是2 (I/Q分量)")
        
        # 验证模式
        if mode not in ['predict', 'evaluate', 'train']:
            raise ValueError(f"不支持的模式: {mode}")
        
        if mode in ['evaluate', 'train'] and labels is None:
            raise ValueError(f"{mode}模式需要提供labels")
        
        # 调用处理函数
        result = process(config, signal, labels, mode)
        
        # 检查处理结果
        if 'error' in result:
            return {
                "result": {},
                "metrics": {},
                "log": f"处理错误: {result['error']}",
                "success": False
            }
        
        # 根据模式构建返回结果
        if mode == 'predict':
            main_result = {
                'detections': result['detections'],
                'detection_values': result['detection_values'],
                'detection_positions': result['detection_positions'],
                'correlation_threshold': result['correlation_threshold'],
                'template_info': result['template_info']
            }
            
            metrics = {}
            
            log_info = []
            log_info.append(f"匹配滤波检测 - 预测模式")
            log_info.append(f"输入信号形状: {signal.shape}")
            log_info.append(f"检测阈值: {result['correlation_threshold']:.4f}")
            log_info.append(f"检测到信号: {np.sum(result['detections'])}/{len(result['detections'])} 个")
            log_info.append(f"模板信息: {result['template_info']['type']}, 长度: {result['template_info']['length']}")
        
        elif mode == 'evaluate':
            main_result = {
                'detections': result['detections'],
                'detection_values': result['detection_values'],
                'detection_positions': result['detection_positions'],
                'correlation_threshold': result['correlation_threshold'],
                'confusion_matrix': result['confusion_matrix'],
                'template_info': result['template_info']
            }
            
            metrics = result['metrics']
            
            log_info = []
            log_info.append(f"匹配滤波检测 - 评估模式")
            log_info.append(f"输入信号形状: {signal.shape}")
            log_info.append(f"检测阈值: {result['correlation_threshold']:.4f}")
            log_info.append("性能指标:")
            log_info.append(f"  检测率: {metrics['detection_rate']:.4f}")
            log_info.append(f"  虚警率: {metrics['false_alarm_rate']:.4f}")
            log_info.append(f"  准确率: {metrics['accuracy']:.4f}")
            log_info.append(f"  召回率: {metrics['recall']:.4f}")
        
        elif mode == 'train':
            main_result = {
                'detections': result['detections'],
                'detection_values': result['detection_values'],
                'detection_positions': result['detection_positions'],
                'correlation_threshold': result['correlation_threshold'],
                'calibrated_threshold': result['calibrated_threshold'],
                'template_info': result['template_info']
            }
            
            metrics = result.get('calibration_metrics', {})
            
            log_info = []
            log_info.append(f"匹配滤波检测 - 训练/校准模式")
            log_info.append(f"输入信号形状: {signal.shape}")
            log_info.append(f"原始阈值: {result['correlation_threshold']:.4f}")
            
            if 'calibrated_threshold' in result:
                log_info.append(f"校准后阈值: {result['calibrated_threshold']:.4f}")
            
            if metrics:
                log_info.append("校准性能:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and key != 'confusion_matrix':
                        log_info.append(f"  {key}: {value:.4f}")
    
        return {
            "result": main_result,
            "metrics": metrics,
            "log": "\n".join(log_info),
            "success": True
        }
        
    except Exception as e:
        return {
            "result": {},
            "metrics": {},
            "log": f"运行错误: {str(e)}",
            "success": False
        }


def build_result_by_mode(result: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """根据不同模式构建返回结果"""
    
    if mode == 'predict':
        # predict模式：只返回检测结果
        main_result = {
            'detections': result.get('detections', []),           # 检测结果 (0/1数组)
            'detection_values': result.get('detection_values', []), # 检测值
            'threshold': result.get('correlation_threshold', 0),    # 使用的阈值
            'detection_positions': result.get('detection_positions', [])  # 检测位置
        }
        
        metrics = {}  # predict模式不返回性能指标
        
        log_info = []
        log_info.append(f"匹配滤波检测 - 预测模式")
        log_info.append(f"输入信号数量: {len(result.get('detections', []))}")
        log_info.append(f"检测阈值: {result.get('correlation_threshold', 0):.4f}")
        
        detections = result.get('detections', [])
        if len(detections) > 0:
            total_detections = np.sum(detections)
            log_info.append(f"检测到信号: {total_detections}/{len(detections)} 个")
        
    elif mode == 'evaluate':
        # evaluate模式：返回检测结果和性能指标
        main_result = {
            'detections': result.get('detections', []),
            'detection_values': result.get('detection_values', []),
            'threshold': result.get('correlation_threshold', 0),
            'confusion_matrix': result.get('confusion_matrix', {}),
            'detection_positions': result.get('detection_positions', [])
        }
        
        metrics = result.get('metrics', {})
        
        log_info = []
        log_info.append(f"匹配滤波检测 - 评估模式")
        log_info.append(f"输入信号数量: {len(result.get('detections', []))}")
        log_info.append(f"检测阈值: {result.get('correlation_threshold', 0):.4f}")
        
        if metrics:
            log_info.append("性能指标:")
            if 'detection_rate' in metrics:
                log_info.append(f"  检测率: {metrics['detection_rate']:.4f}")
            if 'false_alarm_rate' in metrics:
                log_info.append(f"  虚警率: {metrics['false_alarm_rate']:.4f}")
            if 'accuracy' in metrics:
                log_info.append(f"  准确率: {metrics['accuracy']:.4f}")
            if 'precision' in metrics:
                log_info.append(f"  精确率: {metrics['precision']:.4f}")
            if 'recall' in metrics:
                log_info.append(f"  召回率: {metrics['recall']:.4f}")
        
    elif mode == 'train':
        # train模式：返回校准后的参数
        main_result = {
            'calibrated_threshold': result.get('calibrated_threshold', 0),
            'optimal_template': result.get('optimal_template', []),
            'calibration_metrics': result.get('calibration_metrics', {}),
            'detections': result.get('detections', [])  # 校准过程中的检测结果
        }
        
        metrics = result.get('calibration_metrics', {})
        
        log_info = []
        log_info.append(f"匹配滤波检测 - 训练/校准模式")
        log_info.append(f"输入训练信号数量: {len(result.get('detections', []))}")
        
        if 'calibrated_threshold' in result:
            log_info.append(f"校准后阈值: {result['calibrated_threshold']:.4f}")
        
        if metrics:
            log_info.append("校准性能:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_info.append(f"  {key}: {value:.4f}")
    
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    return {
        "result": main_result,
        "metrics": metrics,
        "log": "\n".join(log_info),
        "success": 'error' not in result
    }

# 测试函数
def test():
    """简化的匹配滤波检测算法测试"""
    print("=== 匹配滤波检测算法测试 ===")
    
    # 加载测试数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        test_labels = np.load(os.path.join(data_dir, 'example_labels.npy'))
        print(f"数据加载成功: {test_signal.shape}, 标签分布: {np.bincount(test_labels)}")
    except FileNotFoundError:
        print("未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 测试评估模式
    print(f"\n=== 评估模式测试 ===")
    input_data = {
        "signal": test_signal,
        "labels": test_labels,
        "mode": "evaluate"
    }
    
    result = run(input_data)
    
    if result['success']:
        metrics = result['metrics']
        print(f"检测率: {metrics.get('detection_rate', 0):.3f}")
        print(f"虚警率: {metrics.get('false_alarm_rate', 0):.3f}")
        print(f"准确率: {metrics.get('accuracy', 0):.3f}")
        print(f"召回率: {metrics.get('recall', 0):.3f}")
        
        # 混淆矩阵
        if 'confusion_matrix' in result['result']:
            cm = result['result']['confusion_matrix']
            print(f"混淆矩阵: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
    else:
        print(f"测试失败: {result['log']}")
    
    # 测试训练模式
    print(f"\n=== 训练模式测试 ===")
    input_data['mode'] = 'train'
    
    result = run(input_data)
    
    if result['success']:
        print(f"校准后阈值: {result['result'].get('calibrated_threshold', 0):.3f}")
        if result['metrics']:
            print(f"校准后检测率: {result['metrics'].get('detection_rate', 0):.3f}")
            print(f"校准后虚警率: {result['metrics'].get('false_alarm_rate', 0):.3f}")
    else:
        print(f"训练失败: {result['log']}")
    
    # 测试预测模式
    print(f"\n=== 预测模式测试 ===")
    input_data = {
        "signal": test_signal[:10],  # 只测试前10个样本
        "mode": "predict"
    }
    
    result = run(input_data)
    
    if result['success']:
        detections = result['result']['detections']
        print(f"预测结果: {detections}")
        print(f"检测到信号: {np.sum(detections)}/{len(detections)} 个")
    else:
        print(f"预测失败: {result['log']}")

if __name__ == "__main__":
    test()
