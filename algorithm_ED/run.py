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
        
        if mode not in ['predict', 'evaluate']:
            raise ValueError(f"不支持的运行模式: {mode}，支持的模式: predict, evaluate")
        
        # 验证evaluate模式需要标签
        if mode == 'evaluate' and labels is None:
            raise ValueError("评估模式需要提供标签数据")
        
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
        import traceback
        error_details = traceback.format_exc()
        
        # 异常处理
        return {
            "result": None,
            "metrics": {},
            "log": f"能量检测算法执行失败: {str(e)}\n错误详情:\n{error_details}",
            "success": False
        }

def test():
    """详细的能量检测算法测试 - 用于展示"""
    print("=" * 80)
    print("                     能量检测频谱感知算法测试报告")
    print("=" * 80)
    print(f"算法名称: 能量检测 (Energy Detection)")
    print(f"算法版本: v1.0")
    print(f"测试时间: {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}")
    print(f"测试环境: Python {__import__('sys').version.split()[0]}")
    print("=" * 80)
    
    # 加载测试数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        test_labels = np.load(os.path.join(data_dir, 'example_labels.npy'))
        print("\n1. 数据加载阶段")
        print("-" * 40)
        print(f"✓ 数据加载成功")
        print(f"  - 输入信号形状: {test_signal.shape}")
        print(f"  - 信号数据类型: {test_signal.dtype}")
        print(f"  - 信号数值范围: [{test_signal.min():.4f}, {test_signal.max():.4f}]")
        print(f"  - 标签形状: {test_labels.shape}")
        print(f"  - 标签数据类型: {test_labels.dtype}")
        print(f"  - 标签分布: 无信号={np.sum(test_labels==0)}, 有信号={np.sum(test_labels==1)}")
        print(f"  - 信号占比: {np.mean(test_labels):.1%}")
        
        # 显示配置信息
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\n2. 配置参数")
        print("-" * 40)
        for key, value in config.items():
            print(f"  - {key}: {value}")
            
    except FileNotFoundError:
        print("❌ 未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 测试predict模式
    print(f"\n3. 预测模式测试")
    print("=" * 40)
    
    input_data = {
        "signal": test_signal,
        "mode": "predict"
    }
    
    print("输入数据详情:")
    print(f"  - signal.shape: {input_data['signal'].shape}")
    print(f"  - mode: {input_data['mode']}")
    print(f"  - 第一个样本前10个I分量: {input_data['signal'][0, :10, 0]}")
    print(f"  - 第一个样本前10个Q分量: {input_data['signal'][0, :10, 1]}")
    
    print("\n执行预测...")
    result = run(input_data)
    
    print("\n输出结果详情:")
    if result['success']:
        predictions = result['result']
        print(f"✓ 算法执行成功")
        print(f"  - success: {result['success']}")
        print(f"  - result.shape: {predictions.shape}")
        print(f"  - result.dtype: {predictions.dtype}")
        print(f"  - 检测到信号的样本数: {np.sum(predictions)}/{len(predictions)} ({np.mean(predictions):.1%})")
        print(f"  - metrics: {result['metrics']}")
        
        print(f"\n完整预测结果:")
        print(f"  [", end="")
        for i, pred in enumerate(predictions):
            if i > 0 and i % 20 == 0:
                print(f"\n   ", end="")
            print(f"{pred}", end="")
            if i < len(predictions) - 1:
                print(", ", end="")
        print("]")
        
        print(f"\n算法日志:")
        log_lines = result['log'].split('\n')
        for line in log_lines:
            if line.strip():
                print(f"  {line}")
                
    else:
        print(f"❌ 预测失败")
        print(f"  错误信息: {result['log']}")
    
    # 测试evaluate模式
    print(f"\n\n4. 评估模式测试")
    print("=" * 40)
    
    input_data = {
        "signal": test_signal,
        "labels": test_labels,
        "mode": "evaluate"
    }
    
    print("输入数据详情:")
    print(f"  - signal.shape: {input_data['signal'].shape}")
    print(f"  - labels.shape: {input_data['labels'].shape}")
    print(f"  - mode: {input_data['mode']}")
    print(f"  - 真实标签分布: {dict(zip(*np.unique(input_data['labels'], return_counts=True)))}")
    
    print("\n执行评估...")
    result = run(input_data)
    
    print("\n输出结果详情:")
    if result['success']:
        predictions = result['result']
        metrics = result['metrics']
        
        print(f"✓ 算法执行成功")
        print(f"  - success: {result['success']}")
        print(f"  - result.shape: {predictions.shape}")
        print(f"  - result.dtype: {predictions.dtype}")
        
        # 详细性能指标
        print(f"\n性能指标详情:")
        print(f"  核心指标:")
        print(f"    - 准确率 (Accuracy): {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
        print(f"    - 检测率 (Detection Rate): {metrics.get('detection_rate', 0):.4f} ({metrics.get('detection_rate', 0)*100:.2f}%)")
        print(f"    - 虚警率 (False Alarm Rate): {metrics.get('false_alarm_rate', 0):.4f} ({metrics.get('false_alarm_rate', 0)*100:.2f}%)")
        print(f"    - 召回率 (Recall): {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)")
        print(f"    - 精确率 (Precision): {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)")
        
        print(f"  混淆矩阵:")
        tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
        print(f"    - 真正例 (TP): {tp}")
        print(f"    - 真负例 (TN): {tn}")
        print(f"    - 假正例 (FP): {fp}")
        print(f"    - 假负例 (FN): {fn}")
        print(f"    - 总样本数: {tp + tn + fp + fn}")
        
        # 混淆矩阵可视化
        print(f"\n  混淆矩阵表格:")
        print(f"                 预测结果")
        print(f"              无信号  有信号")
        print(f"    真实 无信号   {tn:4d}   {fp:4d}")
        print(f"    标签 有信号   {fn:4d}   {tp:4d}")
        
        # 预测结果与真实标签对比
        print(f"\n完整预测结果与真实标签对比:")
        print(f"样本序号 | 真实标签 | 预测结果 | 是否正确")
        print(f"-" * 45)
        for i in range(len(predictions)):
            correct = "✓" if predictions[i] == test_labels[i] else "✗"
            label_name = "有信号" if test_labels[i] == 1 else "无信号"
            pred_name = "有信号" if predictions[i] == 1 else "无信号"
            print(f"{i:6d}   |   {test_labels[i]:1d}({label_name})  |   {predictions[i]:1d}({pred_name})  |   {correct}")
        
        # 分类统计
        correct_predictions = np.sum(predictions == test_labels)
        print(f"\n分类统计:")
        print(f"  - 正确预测样本数: {correct_predictions}/{len(predictions)}")
        print(f"  - 错误预测样本数: {len(predictions) - correct_predictions}/{len(predictions)}")
        print(f"  - 正确率: {correct_predictions/len(predictions):.4f} ({correct_predictions/len(predictions)*100:.2f}%)")
        
        print(f"\n算法执行日志:")
        log_lines = result['log'].split('\n')
        for line in log_lines:
            if line.strip():
                print(f"  {line}")
                
    else:
        print(f"❌ 评估失败")
        print(f"  错误信息: {result['log']}")
    
    # 总结
    print(f"\n\n5. 测试总结")
    print("=" * 40)
    print(f"✓ 算法类型: 传统能量检测算法")
    print(f"✓ 支持模式: 预测模式、评估模式")
    print(f"✓ 阈值方法: 噪声底阈值计算")
    print(f"✓ 输入格式: I/Q复数信号")
    print(f"✓ 输出格式: 二元检测结果")
    if 'metrics' in locals() and result['success']:
        print(f"✓ 算法性能: 检测率={metrics.get('detection_rate', 0):.1%}, 虚警率={metrics.get('false_alarm_rate', 0):.1%}")
    print("=" * 80)
    print("                        测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test()
