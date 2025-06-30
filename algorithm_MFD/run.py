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
            - signal: np.ndarray - 输入信号 (N, signal_length) 或 (N, 2, signal_length)
            - labels: np.ndarray - 标签数据 (N,) - 0表示无信号，1表示有信号 (evaluate模式需要)
            - mode: str - 运行模式：
                * "predict": 预测模式 - 对输入信号进行检测，输出检测结果
                * "evaluate": 评估模式 - 需要labels，计算性能指标
                * "train": 训练模式 - 对传统算法，用于参数校准（很少使用）
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
                "noise_variance": 1.0
            }
        
        # 验证输入数据
        if not isinstance(input_data, dict):
            raise ValueError("input_data必须是字典类型")
        
        if 'signal' not in input_data:
            raise ValueError("input_data必须包含'signal'字段")
        
        signal = input_data['signal']
        labels = input_data.get('labels', None)
        mode = input_data.get('mode', 'predict')  # 默认使用predict模式
        
        # 验证信号格式
        if not isinstance(signal, np.ndarray):
            raise ValueError("signal必须是numpy数组")
        
        # 根据模式验证输入
        if mode == 'evaluate' and labels is None:
            raise ValueError("evaluate模式需要提供labels")
        
        if mode == 'train':
            # 对于传统通信算法，train模式主要用于参数校准
            if labels is None:
                raise ValueError("train模式需要提供labels用于参数校准")
        
        # 调用核心处理函数
        result = process(config, signal, labels, mode)
        
        # 根据模式构建返回结果
        return_result = build_result_by_mode(result, mode)
        
        return return_result
        
    except Exception as e:
        error_msg = f"匹配滤波检测算法运行失败: {str(e)}"
        return {
            "result": {},
            "metrics": {},
            "log": error_msg,
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
    """使用挑战性数据测试匹配滤波检测算法"""
    print("=== 匹配滤波检测算法测试（使用挑战性数据） ===")
    
    # 加载生成的测试数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        test_labels = np.load(os.path.join(data_dir, 'example_labels.npy'))
        print(f"加载数据成功:")
        print(f"  信号形状: {test_signal.shape}")
        print(f"  标签分布: {np.bincount(test_labels)}")
        print(f"  信号功率范围: {np.var(test_signal, axis=1).min():.4f} - {np.var(test_signal, axis=1).max():.4f}")
    except FileNotFoundError:
        print("未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 测试不同的阈值
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"\n=== 测试不同阈值的性能 ===")
    
    for threshold in thresholds:
        print(f"\n--- 阈值 {threshold} ---")
        
        # 修改配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        config = {
            "threshold": threshold,
            "template_type": "bpsk",
            "samples_per_bit": 8
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 测试 evaluate 模式
        input_data_evaluate = {
            "signal": test_signal,
            "labels": test_labels,
            "mode": "evaluate"
        }
        
        result_evaluate = run(input_data_evaluate, config_path)
        
        if result_evaluate['success'] and result_evaluate['metrics']:
            metrics = result_evaluate['metrics']
            print(f"检测率: {metrics.get('detection_rate', 0):.4f}")
            print(f"虚警率: {metrics.get('false_alarm_rate', 0):.4f}")
            print(f"准确率: {metrics.get('accuracy', 0):.4f}")
            print(f"F1分数: {metrics.get('f1_score', 0):.4f}")
            
            # 显示混淆矩阵
            if 'confusion_matrix' in result_evaluate['result']:
                cm = result_evaluate['result']['confusion_matrix']
                print(f"混淆矩阵: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
        else:
            print("测试失败!")
    
    # 测试 train 模式（自动寻找最优阈值）
    print(f"\n=== 测试训练模式（自动阈值校准） ===")
    
    input_data_train = {
        "signal": test_signal,
        "labels": test_labels,
        "mode": "train"
    }
    
    result_train = run(input_data_train)
    
    if result_train['success']:
        print(f"原始阈值: {result_train['result'].get('correlation_threshold', 0):.4f}")
        print(f"校准后阈值: {result_train['result'].get('calibrated_threshold', 0):.4f}")
        
        if 'calibration_metrics' in result_train['result']:
            cal_metrics = result_train['result']['calibration_metrics']
            print(f"校准后性能:")
            print(f"  检测率: {cal_metrics.get('detection_rate', 0):.4f}")
            print(f"  虚警率: {cal_metrics.get('false_alarm_rate', 0):.4f}")
            print(f"  准确率: {cal_metrics.get('accuracy', 0):.4f}")
            print(f"  F1分数: {cal_metrics.get('f1_score', 0):.4f}")
    else:
        print("训练模式测试失败!")
    
    # 分析检测值分布
    print(f"\n=== 分析检测值分布 ===")
    
    input_data_predict = {
        "signal": test_signal,
        "mode": "predict"
    }
    
    result_predict = run(input_data_predict)
    
    if result_predict['success']:
        detection_values = result_predict['result']['detection_values']
        detection_values = np.array(detection_values)
        
        # 分析有信号和无信号样本的检测值
        signal_values = detection_values[test_labels == 1]
        noise_values = detection_values[test_labels == 0]
        
        print(f"有信号样本检测值统计:")
        print(f"  均值: {np.mean(signal_values):.4f}")
        print(f"  标准差: {np.std(signal_values):.4f}")
        print(f"  最小值: {np.min(signal_values):.4f}")
        print(f"  最大值: {np.max(signal_values):.4f}")
        
        print(f"无信号样本检测值统计:")
        print(f"  均值: {np.mean(noise_values):.4f}")
        print(f"  标准差: {np.std(noise_values):.4f}")
        print(f"  最小值: {np.min(noise_values):.4f}")
        print(f"  最大值: {np.max(noise_values):.4f}")
        
        # 计算重叠度
        overlap = np.sum((signal_values < np.max(noise_values)) & (signal_values > np.min(noise_values)))
        print(f"检测值重叠样本数: {overlap}/{len(signal_values)}")
        
        # 建议阈值
        suggested_threshold = (np.mean(signal_values) + np.mean(noise_values)) / 2
        print(f"建议阈值: {suggested_threshold:.4f}")

if __name__ == "__main__":
    test()
