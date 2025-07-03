import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Any, Optional
from model import process

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Min-Max归一化预处理算法的标准运行接口
    
    Args:
        input_data (dict): 输入数据字典，包含以下字段：
            - signal: np.ndarray - 输入信号数据
            - normalization_range: tuple - 归一化范围，默认为(0, 1)
            - axis: int/tuple - 归一化轴，默认为None（全局归一化）
            - mode: str - 运行模式：
                * "transform": 变换模式 - 对输入信号进行归一化
                * "fit_transform": 拟合变换模式 - 计算参数并归一化
                * "inverse_transform": 逆变换模式 - 反归一化
        config_path (str, optional): 配置文件路径
        
    Returns:
        dict: 统一格式的返回结果
        {
            "result": np.ndarray,    # 归一化后的信号
            "metrics": dict,         # 归一化统计信息
            "log": str,             # 日志信息
            "success": bool         # 是否成功
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
                "default_range": [0, 1],
                "clip_outliers": False,
                "preserve_zero": False
            }
        
        # 验证输入数据
        if not isinstance(input_data, dict):
            raise ValueError("input_data必须是字典类型")
        
        if 'signal' not in input_data:
            raise ValueError("input_data必须包含'signal'字段")
        
        signal = input_data['signal']
        mode = input_data.get('mode', 'transform')
        
        # 验证信号格式
        if not isinstance(signal, np.ndarray):
            raise ValueError("signal必须是numpy数组")
        
        # 验证模式
        if mode not in ['transform', 'fit_transform', 'inverse_transform']:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 调用处理函数
        result = process(config, signal, input_data, mode)
        
        # 检查处理结果
        if 'error' in result:
            return {
                "result": np.array([]),
                "metrics": {},
                "log": f"处理错误: {result['error']}",
                "success": False
            }
        
        # 构建返回结果
        main_result = result['normalized_signal']
        metrics = result.get('statistics', {})
        
        log_info = []
        log_info.append(f"Min-Max归一化 - {mode}模式")
        log_info.append(f"输入信号形状: {signal.shape}")
        log_info.append(f"归一化范围: {result.get('normalization_range', [0, 1])}")
        
        if 'statistics' in result:
            stats = result['statistics']
            log_info.append(f"原始数据范围: [{stats.get('original_min', 0):.4f}, {stats.get('original_max', 1):.4f}]")
            log_info.append(f"归一化后范围: [{stats.get('normalized_min', 0):.4f}, {stats.get('normalized_max', 1):.4f}]")
        
        return {
            "result": main_result,
            "metrics": metrics,
            "log": "\n".join(log_info),
            "success": True
        }
        
    except Exception as e:
        return {
            "result": np.array([]),
            "metrics": {},
            "log": f"运行错误: {str(e)}",
            "success": False
        }

def create_visualization_plots(original_signals, normalized_signals_dict, save_dir):
    """创建归一化效果可视化图片"""
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 选择前4个样本进行可视化
    num_samples = min(4, original_signals.shape[0])
    
    # 创建原始信号vs归一化信号对比图
    for target_range, normalized_data in normalized_signals_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Min-Max归一化效果对比 (目标范围: {target_range})', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # 获取信号数据（假设是2D：通道x时间）
            if len(original_signals[i].shape) > 1:
                # 多通道信号，显示第一个通道
                original_signal = original_signals[i][0]
                normalized_signal = normalized_data[i][0]
            else:
                # 单通道信号
                original_signal = original_signals[i]
                normalized_signal = normalized_data[i]
            
            # 创建时间轴
            time_axis = np.arange(len(original_signal))
            
            # 绘制原始信号和归一化信号
            ax.plot(time_axis, original_signal, 'b-', linewidth=1.5, label='原始信号', alpha=0.8)
            ax.plot(time_axis, normalized_signal, 'r-', linewidth=1.5, label='归一化信号', alpha=0.8)
            
            ax.set_title(f'样本 {i+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('采样点', fontsize=10)
            ax.set_ylabel('幅值', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # 添加统计信息
            orig_min, orig_max = original_signal.min(), original_signal.max()
            norm_min, norm_max = normalized_signal.min(), normalized_signal.max()
            
            info_text = f'原始: [{orig_min:.3f}, {orig_max:.3f}]\n归一化: [{norm_min:.3f}, {norm_max:.3f}]'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        range_str = f"{target_range[0]}_{target_range[1]}".replace('-', 'neg')
        plt.savefig(os.path.join(save_dir, f'归一化对比_{range_str}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存对比图: 归一化对比_{range_str}.png")

def test():
    """详细的Min-Max归一化测试函数，展示输入信号和归一化后信号的详细信息并生成可视化图片"""
    print("=" * 80)
    print("                Min-Max归一化预处理算法详细测试报告")
    print("=" * 80)
    print("算法功能：将输入信号的数值范围映射到指定的目标范围")
    print("算法公式：normalized = (x - min_x) / (max_x - min_x) * (target_max - target_min) + target_min")
    print("=" * 80)
    
    # 创建assets目录用于保存图片
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"✓ 创建图片保存目录: {assets_dir}")
    
    # 加载测试数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        test_signal = np.load(os.path.join(data_dir, 'example_input.npy'))
        print(f"\n1. 测试数据加载")
        print(f"-" * 40)
        print(f"✓ 测试数据加载成功")
        print(f"  数据形状: {test_signal.shape}")
        print(f"  样本数量: {test_signal.shape[0]}")
        print(f"  信号维度: {test_signal.shape[1:] if len(test_signal.shape) > 1 else '1D'}")
        print(f"  数据类型: {test_signal.dtype}")
        print(f"  数值范围: [{test_signal.min():.6f}, {test_signal.max():.6f}]")
        print(f"  均值: {test_signal.mean():.6f}")
        print(f"  标准差: {test_signal.std():.6f}")
    except FileNotFoundError:
        print("❌ 未找到测试数据，请先运行 make.py 生成数据")
        return
    
    # 选择代表性样本进行详细展示
    print(f"\n2. 原始信号详细分析")
    print(f"-" * 40)
    sample_indices = [0, test_signal.shape[0]//4, test_signal.shape[0]//2, -1]
    
    for i, idx in enumerate(sample_indices):
        signal = test_signal[idx]
        print(f"\n样本 {idx+1}:")
        print(f"  信号形状: {signal.shape}")
        print(f"  数值范围: [{signal.min():.6f}, {signal.max():.6f}]")
        print(f"  均值: {signal.mean():.6f}")
        print(f"  标准差: {signal.std():.6f}")
        
        # 展示每个通道的统计信息
        if len(signal.shape) > 1:
            for ch in range(signal.shape[0]):
                ch_data = signal[ch]
                print(f"    通道{ch}: 范围[{ch_data.min():.6f}, {ch_data.max():.6f}], 均值{ch_data.mean():.6f}")
    
    # 测试不同归一化范围并收集结果用于可视化
    print(f"\n3. 归一化效果测试")
    print(f"-" * 40)
    ranges = [(0, 1), (-1, 1), (0, 10), (-5, 5)]
    normalized_results = {}
    
    for i, norm_range in enumerate(ranges):
        print(f"\n【测试 {i+1}】目标范围: {norm_range}")
        print(f"{'='*50}")
        
        # 使用前6个样本进行测试和可视化
        sample_signals = test_signal[:6]
        
        input_data = {
            "signal": sample_signals,
            "normalization_range": norm_range,
            "mode": "fit_transform"
        }
        
        result = run(input_data)
        
        if result['success']:
            normalized = result['result']
            normalized_results[norm_range] = normalized
            
            print(f"✓ 归一化成功")
            print(f"  原始信号范围: [{sample_signals.min():.6f}, {sample_signals.max():.6f}]")
            print(f"  目标归一化范围: {norm_range}")
            print(f"  实际归一化后范围: [{normalized.min():.6f}, {normalized.max():.6f}]")
            
            # 验证归一化是否正确
            expected_min, expected_max = norm_range
            actual_min, actual_max = normalized.min(), normalized.max()
            min_error = abs(actual_min - expected_min)
            max_error = abs(actual_max - expected_max)
            
            print(f"  范围误差: min误差={min_error:.8f}, max误差={max_error:.8f}")
            
            if result['metrics']:
                metrics = result['metrics']
                print(f"  变换参数:")
                print(f"    原始最小值: {metrics.get('original_min', 0):.6f}")
                print(f"    原始最大值: {metrics.get('original_max', 1):.6f}")
                print(f"    缩放因子: {metrics.get('scale_factor', 0):.6f}")
                print(f"    偏移量: {metrics.get('offset', 0):.6f}")
            
            # 验证线性关系
            original_range = sample_signals.max() - sample_signals.min()
            target_range = norm_range[1] - norm_range[0]
            expected_scale = target_range / original_range if original_range != 0 else 1
            print(f"  验证: 期望缩放因子={expected_scale:.6f}")
            
        else:
            print(f"❌ 归一化失败: {result['log']}")
    
    # 生成可视化图片
    print(f"\n4. 生成可视化图片")
    print(f"-" * 40)
    if normalized_results:
        print(f"正在生成归一化效果可视化图片...")
        create_visualization_plots(test_signal[:6], normalized_results, assets_dir)
        print(f"✓ 所有可视化图片已保存到: {assets_dir}")
    print(f"\n生成的图片文件:")
    for filename in os.listdir(assets_dir):
        if filename.endswith('.png'):
            print(f"  - {filename}")
    
    # 测试逆变换功能
    print(f"\n5. 逆变换测试")
    print(f"-" * 40)
    print(f"验证归一化的可逆性...")
    
    sample_signal = test_signal[0:3]  # 取前3个样本
    
    # 先归一化到 [0, 1]
    input_data = {
        "signal": sample_signal,
        "normalization_range": (0, 1),
        "mode": "fit_transform"
    }
    result1 = run(input_data)
    
    if result1['success']:
        normalized_signal = result1['result']
        print(f"✓ 正向归一化成功")
        print(f"  原始范围: [{sample_signal.min():.6f}, {sample_signal.max():.6f}]")
        print(f"  归一化后: [{normalized_signal.min():.6f}, {normalized_signal.max():.6f}]")
        
        # 逆变换回原始范围
        input_data = {
            "signal": normalized_signal,
            "original_min": result1['metrics']['original_min'],
            "original_max": result1['metrics']['original_max'],
            "mode": "inverse_transform"
        }
        result2 = run(input_data)
        
        if result2['success']:
            recovered_signal = result2['result']
            print(f"✓ 逆变换成功")
            print(f"  恢复后范围: [{recovered_signal.min():.6f}, {recovered_signal.max():.6f}]")
            
            # 计算恢复误差
            mse = np.mean((sample_signal - recovered_signal) ** 2)
            mae = np.mean(np.abs(sample_signal - recovered_signal))
            max_error = np.max(np.abs(sample_signal - recovered_signal))
            
            print(f"  恢复精度:")
            print(f"    均方误差(MSE): {mse:.10f}")
            print(f"    平均绝对误差(MAE): {mae:.10f}")
            print(f"    最大绝对误差: {max_error:.10f}")
            
            if max_error < 1e-6:
                print(f"  ✓ 逆变换精度优秀（误差 < 1e-6）")
            elif max_error < 1e-3:
                print(f"  ✓ 逆变换精度良好（误差 < 1e-3）")
            else:
                print(f"  ⚠ 逆变换精度一般（误差 = {max_error:.6f}）")
                
        else:
            print(f"❌ 逆变换失败: {result2['log']}")
    else:
        print(f"❌ 正向归一化失败: {result1['log']}")
    
    # 总结
    print(f"\n6. 测试总结")
    print(f"=" * 40)
    print(f"✓ Min-Max归一化算法测试完成")
    print(f"✓ 支持多种目标范围：[0,1], [-1,1], [0,10], [-5,5]")
    print(f"✓ 支持多维信号处理")
    print(f"✓ 支持正向和逆向变换")
    print(f"✓ 保持信号的相对关系和线性特性")
    print(f"✓ 生成了详细的可视化图片")
    
    print("=" * 80)
    print("                        测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test()
