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
    """测试函数"""
    print("=== 匹配滤波检测算法测试 ===")
    
    # 生成测试数据
    N = 50
    signal_length = 128
    
    # 生成测试信号 - 复数信号
    np.random.seed(42)
    test_signal = 0.5 * (np.random.randn(N, signal_length) + 1j * np.random.randn(N, signal_length))
    
    # 添加已知BPSK信号到前25个样本中
    # 使用简单的BPSK序列 [1, -1, 1, 1, -1, -1, 1, -1]
    template_bits = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    samples_per_bit = 8
    signal_template = np.repeat(template_bits, samples_per_bit).astype(complex)
    
    for i in range(25):  # 前25个样本包含信号
        if signal_length >= len(signal_template):
            start_pos = np.random.randint(0, signal_length - len(signal_template) + 1)
            # 添加信号，强度为2.0
            test_signal[i, start_pos:start_pos+len(signal_template)] += 2.0 * signal_template
    
    # 生成标签：前25个有信号，后25个无信号
    test_labels = np.zeros(N)
    test_labels[:25] = 1
    
    # 测试1: predict模式
    print("\n--- 测试 predict 模式 ---")
    input_data_predict = {
        "signal": test_signal,
        "mode": "predict"
    }
    
    result_predict = run(input_data_predict)
    print(f"Predict模式 - 成功: {result_predict['success']}")
    print(f"检测到的信号数量: {np.sum(result_predict['result']['detections'])}")
    print(f"日志:\n{result_predict['log']}")
    
    # 测试2: evaluate模式
    print("\n--- 测试 evaluate 模式 ---")
    input_data_evaluate = {
        "signal": test_signal,
        "labels": test_labels,
        "mode": "evaluate"
    }
    
    result_evaluate = run(input_data_evaluate)
    print(f"Evaluate模式 - 成功: {result_evaluate['success']}")
    print(f"日志:\n{result_evaluate['log']}")
    
    if result_evaluate['success'] and result_evaluate['metrics']:
        print(f"\n性能指标:")
        metrics = result_evaluate['metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    # 测试3: train模式（参数校准）
    print("\n--- 测试 train 模式 ---")
    input_data_train = {
        "signal": test_signal,
        "labels": test_labels,
        "mode": "train"
    }
    
    result_train = run(input_data_train)
    print(f"Train模式 - 成功: {result_train['success']}")
    print(f"日志:\n{result_train['log']}")
    
    if result_train['success']:
        calibrated_threshold = result_train['result'].get('calibrated_threshold', 0)
        print(f"校准后的阈值: {calibrated_threshold:.4f}")

if __name__ == "__main__":
    test()
