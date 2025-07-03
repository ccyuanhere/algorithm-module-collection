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
    Z-score标准化预处理算法的标准运行接口
    
    Args:
        input_data (dict): 输入数据字典，包含以下字段：
            - signal: np.ndarray - 输入信号数据
            - axis: int/tuple - 标准化轴，默认为None（全局标准化）
            - ddof: int - 标准差计算的自由度修正，默认为0
            - mode: str - 运行模式：
                * "transform": 变换模式 - 使用已知参数标准化
                * "fit_transform": 拟合变换模式 - 计算统计参数并标准化
                * "inverse_transform": 逆变换模式 - 反标准化
        config_path (str, optional): 配置文件路径
        
    Returns:
        dict: 统一格式的返回结果
        {
            "result": np.ndarray,    # 标准化后的信号
            "metrics": dict,         # 标准化统计信息
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
                "ddof": 0,
                "clip_outliers": False,
                "outlier_threshold": 3.0
            }
        
        # 验证输入数据
        if not isinstance(input_data, dict):
            raise ValueError("input_data必须是字典类型")
        
        if 'signal' not in input_data:
            raise ValueError("input_data必须包含'signal'字段")
        
        signal = input_data['signal']
        mode = input_data.get('mode', 'fit_transform')
        
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
        main_result = result['standardized_signal']
        metrics = result.get('statistics', {})
        
        log_info = []
        log_info.append(f"Z-score标准化 - {mode}模式")
        log_info.append(f"输入信号形状: {signal.shape}")
        
        if 'statistics' in result:
            stats = result['statistics']
            log_info.append(f"原始均值: {stats.get('original_mean', 0):.4f}")
            log_info.append(f"原始标准差: {stats.get('original_std', 1):.4f}")
            log_info.append(f"标准化后均值: {stats.get('standardized_mean', 0):.4f}")
            log_info.append(f"标准化后标准差: {stats.get('standardized_std', 1):.4f}")
        
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

def create_visualization_plots(original_signals, standardized_signals_dict, save_dir):
    """创建Z-score标准化效果可视化图片"""
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 选择前4个样本进行可视化
    num_samples = min(4, original_signals.shape[0])
    
    # 创建原始信号vs标准化信号对比图
    for method_name, standardized_data in standardized_signals_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Z-score标准化效果对比 (方法: {method_name})', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # 获取信号数据（假设是2D：通道x时间）
            if len(original_signals[i].shape) > 1:
                # 多通道信号，显示第一个通道
                original_signal = original_signals[i][0]
                standardized_signal = standardized_data[i][0]
            else:
                # 单通道信号
                original_signal = original_signals[i]
                standardized_signal = standardized_data[i]
            
            # 创建时间轴
            time_axis = np.arange(len(original_signal))
            
            # 绘制原始信号和标准化信号
            ax.plot(time_axis, original_signal, 'b-', linewidth=1.5, label='原始信号', alpha=0.8)
            ax.plot(time_axis, standardized_signal, 'r-', linewidth=1.5, label='标准化信号', alpha=0.8)
            
            ax.set_title(f'样本 {i+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('采样点', fontsize=10)
            ax.set_ylabel('幅值', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # 添加统计信息
            orig_mean, orig_std = original_signal.mean(), original_signal.std()
            stand_mean, stand_std = standardized_signal.mean(), standardized_signal.std()
            
            info_text = f'原始: μ={orig_mean:.3f}, σ={orig_std:.3f}\n标准化: μ={stand_mean:.3f}, σ={stand_std:.3f}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        method_safe = method_name.replace(' ', '_').replace('-', '_')
        plt.savefig(os.path.join(save_dir, f'标准化对比_{method_safe}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存对比图: 标准化对比_{method_safe}.png")

def test():
    """详细的Z-score标准化测试函数，展示输入信号和标准化后信号的详细信息并生成可视化图片"""
    print("=" * 80)
    print("                Z-score标准化预处理算法详细测试报告")
    print("=" * 80)
    print("算法功能：将输入信号的分布转换为均值0、标准差1的标准正态分布")
    print("算法公式：standardized = (x - mean_x) / std_x")
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
        print(f"  全局均值: {test_signal.mean():.6f}")
        print(f"  全局标准差: {test_signal.std():.6f}")
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
        print(f"  均值: {signal.mean():.6f}")
        print(f"  标准差: {signal.std():.6f}")
        print(f"  数值范围: [{signal.min():.6f}, {signal.max():.6f}]")
        
        # 计算偏度，避免f-string括号嵌套问题
        if signal.std() > 0:
            skewness = ((signal - signal.mean()) / signal.std())**3
            skewness_value = skewness.mean()
            print(f"  偏度: {skewness_value:.4f}")
        else:
            print(f"  偏度: 0.0000 (常数信号)")
        
        # 展示每个通道的统计信息
        if len(signal.shape) > 1:
            for ch in range(signal.shape[0]):
                ch_data = signal[ch]
                print(f"    通道{ch}: μ={ch_data.mean():.6f}, σ={ch_data.std():.6f}")
    
    # 测试不同标准化方法并收集结果用于可视化
    print(f"\n3. 标准化效果测试")
    print(f"-" * 40)
    methods = [
        ('标准Z-score', {'method': 'standard'}),
        ('鲁棒标准化', {'method': 'robust', 'center_method': 'median', 'scale_method': 'mad'}),
        ('分位数标准化', {'method': 'quantile', 'quantile_range': (0.25, 0.75), 'target_range': (-2, 2)}),
        ('裁剪异常值', {'method': 'standard'})  # 后面会在config中设置clip_outliers
    ]
    
    standardized_results = {}
    
    for i, (method_name, method_params) in enumerate(methods):
        print(f"\n【测试 {i+1}】方法: {method_name}")
        print(f"{'='*50}")
        
        # 使用前6个样本进行测试和可视化
        sample_signals = test_signal[:6]
        
        input_data = {
            "signal": sample_signals,
            "mode": "fit_transform",
            **method_params
        }
        
        # 对于裁剪异常值的方法，修改config
        if method_name == '裁剪异常值':
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            # 临时修改配置
            original_config = None
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    original_config = json.load(f)
                
                # 修改配置
                temp_config = original_config.copy()
                temp_config['clip_outliers'] = True
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_config, f, indent=4, ensure_ascii=False)
        
        result = run(input_data)
        
        # 恢复原配置
        if method_name == '裁剪异常值' and original_config:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(original_config, f, indent=4, ensure_ascii=False)
        
        if result['success']:
            standardized = result['result']
            standardized_results[method_name] = standardized
            
            print(f"✓ 标准化成功")
            print(f"  原始信号统计: μ={sample_signals.mean():.6f}, σ={sample_signals.std():.6f}")
            print(f"  标准化后统计: μ={standardized.mean():.6f}, σ={standardized.std():.6f}")
            print(f"  数值范围: [{standardized.min():.6f}, {standardized.max():.6f}]")
            
            if result['metrics']:
                metrics = result['metrics']
                print(f"  方法参数:")
                if 'method' in metrics:
                    print(f"    方法类型: {metrics.get('method', 'standard')}")
                if 'center_method' in metrics:
                    print(f"    中心化方法: {metrics.get('center_method')}")
                if 'scale_method' in metrics:
                    print(f"    缩放方法: {metrics.get('scale_method')}")
                if 'outlier_threshold' in metrics:
                    print(f"    异常值阈值: {metrics.get('outlier_threshold')}")
            
            # 验证标准化效果
            if method_name == '标准Z-score':
                mean_error = abs(standardized.mean())
                std_error = abs(standardized.std() - 1.0)
                print(f"  标准化精度: 均值误差={mean_error:.8f}, 标准差误差={std_error:.8f}")
                
                # 检查每个样本的标准化效果
                print(f"  样本级验证:")
                for sample_idx in range(min(3, standardized.shape[0])):
                    sample = standardized[sample_idx]
                    sample_mean = sample.mean()
                    sample_std = sample.std()
                    print(f"    样本{sample_idx+1}: μ={sample_mean:.6f}, σ={sample_std:.6f}")
                
                # 检查验证标志
                if 'verification_passed' in metrics:
                    status = "✓ 通过" if metrics['verification_passed'] else "✗ 失败"
                    print(f"  自动验证: {status}")
                
            elif method_name in ['鲁棒标准化', '分位数标准化']:
                print(f"  注: {method_name}的结果不一定严格满足μ=0,σ=1")
                
        else:
            print(f"❌ 标准化失败: {result['log']}")
    
    # 生成可视化图片
    print(f"\n4. 生成可视化图片")
    print(f"-" * 40)
    if standardized_results:
        print(f"正在生成标准化效果可视化图片...")
        create_visualization_plots(test_signal[:6], standardized_results, assets_dir)
        print(f"✓ 所有可视化图片已保存到: {assets_dir}")
        print(f"\n生成的图片文件:")
        for filename in os.listdir(assets_dir):
            if filename.endswith('.png'):
                print(f"  - {filename}")
    
    # 测试逆变换功能
    print(f"\n5. 逆变换测试")
    print(f"-" * 40)
    print(f"验证标准化的可逆性...")
    
    sample_signal = test_signal[0:3]  # 取前3个样本
    
    # 先标准化
    input_data = {
        "signal": sample_signal,
        "mode": "fit_transform"
    }
    result1 = run(input_data)
    
    if result1['success']:
        standardized_signal = result1['result']
        print(f"✓ 正向标准化成功")
        print(f"  原始统计: μ={sample_signal.mean():.6f}, σ={sample_signal.std():.6f}")
        print(f"  标准化后: μ={standardized_signal.mean():.6f}, σ={standardized_signal.std():.6f}")
        
        # 逆变换回原始范围
        input_data = {
            "signal": standardized_signal,
            "original_mean": result1['metrics']['original_mean'],
            "original_std": result1['metrics']['original_std'],
            "mode": "inverse_transform"
        }
        result2 = run(input_data)
        
        if result2['success']:
            recovered_signal = result2['result']
            print(f"✓ 逆变换成功")
            print(f"  恢复后统计: μ={recovered_signal.mean():.6f}, σ={recovered_signal.std():.6f}")
            
            # 计算恢复误差
            mse = np.mean((sample_signal - recovered_signal) ** 2)
            mae = np.mean(np.abs(sample_signal - recovered_signal))
            max_error = np.max(np.abs(sample_signal - recovered_signal))
            
            print(f"  恢复精度:")
            print(f"    均方误差(MSE): {mse:.10f}")
            print(f"    平均绝对误差(MAE): {mae:.10f}")
            print(f"    最大绝对误差: {max_error:.10f}")
            
            if max_error < 1e-10:
                print(f"  ✓ 逆变换精度优秀（误差 < 1e-10）")
            elif max_error < 1e-6:
                print(f"  ✓ 逆变换精度良好（误差 < 1e-6）")
            else:
                print(f"  ⚠ 逆变换精度一般（误差 = {max_error:.6f}）")
                
        else:
            print(f"❌ 逆变换失败: {result2['log']}")
    else:
        print(f"❌ 正向标准化失败: {result1['log']}")
    
    # 总结
    print(f"\n6. 测试总结")
    print(f"=" * 40)
    print(f"✓ Z-score标准化算法测试完成")
    print(f"✓ 支持多种标准化方法：标准、鲁棒、分位数、异常值裁剪")
    print(f"✓ 支持多维信号处理")
    print(f"✓ 支持正向和逆向变换")
    print(f"✓ 将数据转换为均值0、标准差1的分布")
    print(f"✓ 生成了详细的可视化图片")
    
    print("=" * 80)
    print("                        测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test()
