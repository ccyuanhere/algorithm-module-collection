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
            - labels: np.ndarray - 标签数据 (N,) - 0表示无信号，1表示有信号 (evaluate模式需要)
            - mode: str - 运行模式：
                * "predict": 预测模式 - 对输入信号进行检测，输出检测结果
                * "evaluate": 评估模式 - 需要labels，计算性能指标
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
        if mode not in ['predict', 'evaluate']:
            raise ValueError(f"不支持的模式: {mode}，仅支持predict和evaluate模式")
        
        if mode == 'evaluate' and labels is None:
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
            main_result = result['detections']
            
            metrics = {}
            
            log_info = []
            log_info.append("匹配滤波检测 - 预测模式")
            log_info.append(f"输入信号形状: {signal.shape}")
            log_info.append(f"检测阈值: {result['correlation_threshold']:.4f}")
            log_info.append(f"检测到信号数量: {np.sum(result['detections'])}/{len(result['detections'])} 个")
            log_info.append(f"模板类型: {result['template_info']['type']}, 长度: {result['template_info']['length']}")
        
        elif mode == 'evaluate':
            main_result = result['detections']
            
            metrics = result['metrics']
            
            log_info = []
            log_info.append("匹配滤波检测 - 评估模式")
            log_info.append(f"输入信号形状: {signal.shape}")
            log_info.append(f"检测阈值: {result['correlation_threshold']:.4f}")
            log_info.append("算法性能指标:")
            log_info.append(f"  检测率(召回率): {metrics['detection_rate']:.4f}")
            log_info.append(f"  虚警率: {metrics['false_alarm_rate']:.4f}")
            log_info.append(f"  准确率: {metrics['accuracy']:.4f}")
            log_info.append(f"  精确率: {metrics['precision']:.4f}")
            log_info.append(f"  F1分数: {metrics['f1_score']:.4f}")
            
            # 混淆矩阵信息
            cm = result['confusion_matrix']
            log_info.append(f"混淆矩阵: 真正例={cm['tp']}, 真负例={cm['tn']}, 假正例={cm['fp']}, 假负例={cm['fn']}")
            log_info.append(f"模板类型: {result['template_info']['type']}, 长度: {result['template_info']['length']}")
    
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

def test():
    """详细的匹配滤波检测算法测试报告 - 用于展示"""
    print("=" * 80)
    print("                   匹配滤波检测算法测试报告")
    print("=" * 80)
    print(f"算法名称: 匹配滤波检测 (Matched Filter Detection)")
    print(f"算法版本: v1.0")
    print(f"测试时间: {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}")
    print(f"测试环境: Python {__import__('sys').version.split()[0]}")
    print("=" * 80)
    
    # 1. 已知信号模板信息
    print("\n1. 已知信号模板详细信息")
    print("-" * 40)
    try:
        from model import load_known_template, generate_default_template
        template = load_known_template()
        
        print(f"✓ 模板加载成功")
        print(f"  - 模板数据类型: 复数信号 (Complex)")
        print(f"  - 模板长度: {len(template)} 个采样点")
        print(f"  - 模板幅度范围: [{np.abs(template).min():.6f}, {np.abs(template).max():.6f}]")
        print(f"  - 模板相位范围: [{np.angle(template).min():.6f}, {np.angle(template).max():.6f}] 弧度")
        print(f"  - 平均功率: {np.mean(np.abs(template)**2):.8f}")
        print(f"  - 峰值功率: {np.max(np.abs(template)**2):.8f}")
        print(f"  - RMS值: {np.sqrt(np.mean(np.abs(template)**2)):.8f}")
        
        # 显示模板详细数据
        print(f"\n模板完整数据 (前20个采样点):")
        print(f"序号 |     实部     |     虚部     |     幅度     |     相位")
        print(f"-" * 65)
        for i in range(min(20, len(template))):
            real_part = template[i].real
            imag_part = template[i].imag
            magnitude = np.abs(template[i])
            phase = np.angle(template[i])
            print(f"{i:3d}  | {real_part:+11.6f} | {imag_part:+11.6f} | {magnitude:11.6f} | {phase:+10.6f}")
        
        if len(template) > 20:
            print(f"... (省略剩余 {len(template)-20} 个点)")
        
        # 模板类型说明
        print(f"\n模板信号说明:")
        print(f"  - 信号类型: 巴克码 (Barker Code)")
        print(f"  - 编码序列: [1, 1, 1, -1, -1, 1, -1] (长度7)")
        print(f"  - 调制方式: BPSK (Binary Phase Shift Keying)")
        print(f"  - 每符号采样: 8个采样点")
        print(f"  - 总长度: 7 × 8 = 56 个采样点")
        print(f"  - 特点: 具有良好的自相关特性，适合用作同步序列")
        
    except Exception as e:
        print(f"❌ 模板加载失败: {e}")
        return
    
    # 2. 加载和分析测试数据
    print(f"\n2. 测试数据生成和加载")
    print("-" * 40)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        test_labels = np.load(os.path.join(data_dir, 'example_labels.npy'))
        
        print(f"✓ 数据加载成功")
        print(f"  - 输入信号形状: {test_signal.shape}")
        print(f"  - 信号数据类型: {test_signal.dtype}")
        print(f"  - I分量数值范围: [{test_signal[:,:,0].min():.6f}, {test_signal[:,:,0].max():.6f}]")
        print(f"  - Q分量数值范围: [{test_signal[:,:,1].min():.6f}, {test_signal[:,:,1].max():.6f}]")
        print(f"  - 标签形状: {test_labels.shape}")
        print(f"  - 标签数据类型: {test_labels.dtype}")
        print(f"  - 标签分布: 无信号={np.sum(test_labels==0)}, 有信号={np.sum(test_labels==1)}")
        print(f"  - 信号占比: {np.mean(test_labels):.1%}")
        
        # 数据生成说明
        print(f"\n测试数据生成说明:")
        print(f"  - 数据来源: make.py 自动生成")
        print(f"  - 无信号样本: 纯高斯白噪声")
        print(f"  - 有信号样本: 巴克码模板 + 高斯白噪声")
        print(f"  - SNR范围: -5dB 到 +15dB")
        print(f"  - 每个样本长度: {test_signal.shape[1]} 个I/Q采样点")
        print(f"  - 模板在信号中的位置: 随机插入")
        
        # 显示配置信息
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"\n算法配置参数:")
            for key, value in config.items():
                print(f"  - {key}: {value}")
        except:
            print(f"\n使用默认配置参数")
            
    except FileNotFoundError:
        print("❌ 未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 3. 预测模式测试
    print(f"\n\n3. 预测模式测试")
    print("=" * 40)
    
    input_data = {
        "signal": test_signal,
        "mode": "predict"
    }
    
    print("输入数据详情:")
    print(f"  - signal.shape: {input_data['signal'].shape}")
    print(f"  - mode: {input_data['mode']}")
    print(f"  - 第一个样本前5个I分量: {input_data['signal'][0, :5, 0]}")
    print(f"  - 第一个样本前5个Q分量: {input_data['signal'][0, :5, 1]}")
    
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
            if i > 0 and i % 25 == 0:
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
    
    # 4. 评估模式测试
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
        print(f"    - 准确率 (Accuracy): {metrics.get('accuracy', 0):.6f} ({metrics.get('accuracy', 0)*100:.3f}%)")
        print(f"    - 检测率 (Detection Rate): {metrics.get('detection_rate', 0):.6f} ({metrics.get('detection_rate', 0)*100:.3f}%)")
        print(f"    - 虚警率 (False Alarm Rate): {metrics.get('false_alarm_rate', 0):.6f} ({metrics.get('false_alarm_rate', 0)*100:.3f}%)")
        print(f"    - 召回率 (Recall): {metrics.get('recall', 0):.6f} ({metrics.get('recall', 0)*100:.3f}%)")
        print(f"    - 精确率 (Precision): {metrics.get('precision', 0):.6f} ({metrics.get('precision', 0)*100:.3f}%)")
        print(f"    - F1分数 (F1-Score): {metrics.get('f1_score', 0):.6f}")
        
        # 从完整结果中获取混淆矩阵
        full_result_data = process(config, test_signal, test_labels, 'evaluate')
        if 'confusion_matrix' in full_result_data:
            cm = full_result_data['confusion_matrix']
            tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
        else:
            tp = np.sum((predictions == 1) & (test_labels == 1))
            tn = np.sum((predictions == 0) & (test_labels == 0))
            fp = np.sum((predictions == 1) & (test_labels == 0))
            fn = np.sum((predictions == 0) & (test_labels == 1))
        
        print(f"  混淆矩阵:")
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
        
        # 预测结果与真实标签对比 (显示前50个)
        print(f"\n预测结果与真实标签对比 (前50个样本):")
        print(f"样本序号 | 真实标签 | 预测结果 | 是否正确")
        print(f"-" * 45)
        for i in range(min(50, len(predictions))):
            correct = "✓" if predictions[i] == test_labels[i] else "✗"
            label_name = "有信号" if test_labels[i] == 1 else "无信号"
            pred_name = "有信号" if predictions[i] == 1 else "无信号"
            print(f"{i:6d}   |   {test_labels[i]:1d}({label_name})  |   {predictions[i]:1d}({pred_name})  |   {correct}")
        
        if len(predictions) > 50:
            print(f"... (省略剩余 {len(predictions)-50} 个样本)")
        
        # 分类统计
        correct_predictions = np.sum(predictions == test_labels)
        print(f"\n分类统计:")
        print(f"  - 正确预测样本数: {correct_predictions}/{len(predictions)}")
        print(f"  - 错误预测样本数: {len(predictions) - correct_predictions}/{len(predictions)}")
        print(f"  - 正确率: {correct_predictions/len(predictions):.6f} ({correct_predictions/len(predictions)*100:.3f}%)")
        
        print(f"\n算法执行日志:")
        log_lines = result['log'].split('\n')
        for line in log_lines:
            if line.strip():
                print(f"  {line}")
                
    else:
        print(f"❌ 评估失败")
        print(f"  错误信息: {result['log']}")
    
    # 5. 算法性能分析
    print(f"\n\n5. 算法性能深度分析")
    print("=" * 40)
    
    if result['success'] and 'metrics' in result and result['metrics']:
        metrics = result['metrics']
        
        print(f"✓ 匹配滤波检测算法性能评估:")
        print(f"  算法类型: 传统信号检测算法")
        print(f"  检测原理: 基于已知模板的相关匹配")
        print(f"  模板类型: 7位巴克码序列")
        print(f"  适用场景: 已知信号特征的频谱感知")
        
        print(f"\n关键性能指标:")
        dr = metrics.get('detection_rate', 0)
        far = metrics.get('false_alarm_rate', 0)
        acc = metrics.get('accuracy', 0)
        
        # 性能等级评估
        if dr >= 0.9 and far <= 0.1:
            performance_level = "优秀"
        elif dr >= 0.8 and far <= 0.2:
            performance_level = "良好"
        elif dr >= 0.7 and far <= 0.3:
            performance_level = "一般"
        else:
            performance_level = "需要优化"
            
        print(f"  - 综合性能等级: {performance_level}")
        print(f"  - 检测率 vs 虚警率权衡: {'平衡良好' if abs(dr - (1-far)) < 0.3 else '存在偏向'}")
        print(f"  - 模板匹配效果: {'高效' if dr >= 0.8 else '中等' if dr >= 0.6 else '较低'}")
        
        # 与理论性能对比
        print(f"\n与传统能量检测对比:")
        print(f"  - 优势: 利用先验信息，抗噪能力强")
        print(f"  - 劣势: 需要已知信号模板")
        print(f"  - 适用性: 特定信号检测场景")
    
    # 6. 总结
    print(f"\n\n6. 测试总结")
    print("=" * 40)
    print(f"✓ 算法类型: 匹配滤波检测算法")
    print(f"✓ 支持模式: 预测模式、评估模式")
    print(f"✓ 检测方法: 基于已知模板的相关运算")
    print(f"✓ 输入格式: I/Q复数信号")
    print(f"✓ 输出格式: 二元检测结果")
    print(f"✓ 模板信号: 7位巴克码 (56个采样点)")
    if 'metrics' in locals() and result['success']:
        print(f"✓ 算法性能: 检测率={metrics.get('detection_rate', 0):.1%}, 虚警率={metrics.get('false_alarm_rate', 0):.1%}")
    print("=" * 80)
    print("                        测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test()
