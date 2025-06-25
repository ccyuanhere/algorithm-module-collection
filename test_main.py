import os
import sys
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入算法模块
from generator_AIS.run import run as generator_run
from preprocessor_normalize.run import run as preprocessor_run
from algorithm_knn.run import run as knn_run

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 只显示消息，不显示日志级别和时间
)
logger = logging.getLogger(__name__)

def print_result_info(result: Dict[str, Any], module_name: str) -> None:
    """
    打印测试结果信息
    
    Args:
        result: 运行结果字典
        module_name: 模块名称
    """
    logger.info(f"\n=== {module_name}模块运行结果 ===")
    
    # 1. 打印主结果的type和size
    if result["result"] is not None:
        if isinstance(result["result"], dict):
            for key, value in result["result"].items():
                if isinstance(value, np.ndarray):
                    logger.info(f"主结果 {key}:")
                    logger.info(f"  - 类型: {type(value).__name__}")
                    logger.info(f"  - 形状: {value.shape}")
                else:
                    logger.info(f"主结果 {key}: {type(value).__name__}")
        else:
            logger.info(f"主结果类型: {type(result['result']).__name__}")
            if isinstance(result["result"], np.ndarray):
                logger.info(f"主结果形状: {result['result'].shape}")
    
    # 2. 打印可选评估值
    logger.info("\n评估指标:")
    if result["metrics"]:
        for key, value in result["metrics"].items():
            if isinstance(value, np.ndarray):
                logger.info(f"  - {key}: 形状 {value.shape}")
            else:
                logger.info(f"  - {key}: {value}")
    else:
        logger.info("  - 无评估指标")
    
    # 3. 打印日志信息
    logger.info(f"\n日志信息: {result['log']}")
    
    # 4. 打印执行成功标志
    logger.info(f"\n执行状态: {'成功' if result['success'] else '失败'}")

def test_single_module(module_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    测试单个算法模块
    
    Args:
        module_name: 模块文件夹名称（'generator_AIS', 'preprocessor_normalize', 'algorithm_knn'）
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (测试是否通过, 运行结果)
    """
    try:
        logger.info(f"\n=== 测试{module_name}模块 ===")
        
        # 第1步：准备输入参数和数据
        if module_name == 'generator_AIS':
            # 生成器需要完整的参数配置
            input_data = {
                "signal_length": 520,
                "modulation_params": {
                    "snr_db": 20,
                    "modulation_types": ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"],
                    "samples_per_type": 10
                }
            }
        elif module_name == 'preprocessor_normalize':
            # 从预处理器模块的data文件夹加载测试数据
            data_dir = current_dir / "preprocessor_normalize" / "data"
            try:
                signal = np.load(data_dir / "example_input.npy")
                logger.info(f"成功加载预处理器测试数据: {signal.shape}")
                input_data = {
                    "signal": signal
                }
            except Exception as e:
                raise RuntimeError(f"加载预处理器测试数据失败: {str(e)}")
        elif module_name == 'algorithm_knn':
            # 从KNN模块的data文件夹加载测试数据
            data_dir = current_dir / "algorithm_knn" / "data"
            try:
                signal = np.load(data_dir / "example_input.npy")
                labels = np.load(data_dir / "example_labels.npy")
                logger.info(f"成功加载KNN测试数据: {signal.shape} 信号, {len(labels)} 标签")
                input_data = {
                    "signal": signal,
                    "labels": labels,
                    "mode": "train"
                }
            except Exception as e:
                raise RuntimeError(f"加载KNN测试数据失败: {str(e)}")
        
        # 第2步：调用run函数
        if module_name == 'generator_AIS':
            result = generator_run(input_data)
        elif module_name == 'preprocessor_normalize':
            result = preprocessor_run(input_data)
        elif module_name == 'algorithm_knn':
            result = knn_run(input_data)
        
        # 第3步：打印结果信息
        print_result_info(result, module_name)
        
        return result["success"], result
        
    except Exception as e:
        error_msg = f"{module_name}模块测试出错: {str(e)}"
        logger.error(error_msg)
        return False, {
            "result": None,
            "metrics": {},
            "log": error_msg,
            "success": False
        }


def test_pipeline() -> Tuple[bool, Dict[str, Any]]:
    """
    简化版完整流程测试：生成器 → 预处理器 → KNN算法
    """
    try:
        logger.info("\n=== 测试完整处理流程 ===")

        # 步骤1：生成信号
        logger.info("\n1. 生成信号")
        gen_input = {
            "signal_length": 520,
            "modulation_params": {
                "snr_db": 20,
                "modulation_types": ["BPSK", "QPSK", "8PSK", "16QAM"],
                "samples_per_type": 20
            }
        }

        gen_result = generator_run(gen_input)
        if not gen_result["success"]:
            logger.error("信号生成失败")
            return False, gen_result

        # 获取生成的信号 - 处理字典结果
        if isinstance(gen_result["result"], dict):
            # 如果result是字典，尝试获取信号数据
            if "signals" in gen_result["result"]:
                generated_signals = gen_result["result"]["signals"]
            elif "signal" in gen_result["result"]:
                generated_signals = gen_result["result"]["signal"]
            else:
                # 打印结果结构帮助调试
                logger.error(f"生成器结果结构: {list(gen_result['result'].keys())}")
                return False, {"success": False, "error": "无法从生成器结果中提取信号"}
        else:
            # 如果result直接是数组
            generated_signals = gen_result["result"]

        # 根据配置文件，只有1种调制方式（QPSK），100个样本
        generated_labels = np.tile([0, 1, 2, 3], len(generated_signals) // 4 + 1)[:len(generated_signals)]
        logger.info(f"创建多类别标签，分布: {np.bincount(generated_labels)}")
        logger.info(f"信号生成成功: {generated_signals.shape}, 标签: {len(generated_labels)}个")

        # 步骤2：预处理信号
        logger.info("\n2. 预处理信号")
        preprocess_input = {
            "signal": generated_signals
        }

        preprocess_result = preprocessor_run(preprocess_input)
        if not preprocess_result["success"]:
            logger.error("信号预处理失败")
            return False, preprocess_result

        preprocessed_signals = preprocess_result["result"]
        logger.info(f"预处理成功: {preprocessed_signals.shape}")

        # 步骤3：KNN训练
        logger.info("\n3. KNN训练")
        knn_input = {
            "signal": preprocessed_signals,
            "labels": generated_labels,
            "mode": "train"
        }

        knn_result = knn_run(knn_input)
        if not knn_result["success"]:
            logger.error("KNN训练失败")
            return False, knn_result

        logger.info("KNN训练成功")

        # 简单的成功结果
        return True, {
            "result": {
                "signal_shape": generated_signals.shape,
                "preprocessed_shape": preprocessed_signals.shape,
                "accuracy": knn_result["metrics"].get("accuracy", "N/A")
            },
            "metrics": {
                "total_samples": len(generated_signals),
                "accuracy": knn_result["metrics"].get("accuracy", 0)
            },
            "log": "完整流程测试成功",
            "success": True
        }

    except Exception as e:
        error_msg = f"流程测试出错: {str(e)}"
        logger.error(error_msg)
        return False, {
            "result": None,
            "metrics": {},
            "log": error_msg,
            "success": False
        }

def main():
    """
    主函数：运行所有测试
    """
    # 设置随机种子，确保结果可重复
    np.random.seed(42)

    # 测试结果统计
    total_tests = 4  # 总共4个测试
    passed_tests = 0

    try:
        # 1. 测试信号生成器
        logger.info("=== 开始测试信号生成器 ===")
        gen_success, _ = test_single_module('generator_AIS')
        if gen_success:
            passed_tests += 1
            logger.info("✅ 信号生成器测试通过")
        else:
            logger.error("❌ 信号生成器测试失败")

        # 2. 测试预处理器
        logger.info("\n=== 开始测试预处理器 ===")
        preprocess_success, _ = test_single_module('preprocessor_normalize')
        if preprocess_success:
            passed_tests += 1
            logger.info("✅ 预处理器测试通过")
        else:
            logger.error("❌ 预处理器测试失败")

        # 3. 测试KNN算法
        logger.info("\n=== 开始测试KNN算法 ===")
        knn_success, _ = test_single_module('algorithm_knn')
        if knn_success:
            passed_tests += 1
            logger.info("✅ KNN算法测试通过")
        else:
            logger.error("❌ KNN算法测试失败")

        # 4. 测试完整流程
        logger.info("\n=== 开始测试完整流程 ===")
        success, result = test_pipeline()
        if success:
            passed_tests += 1
            logger.info("✅ 完整流程测试通过")
            if result and "result" in result:
                # 修改这里：适配简化版的结果结构
                if "accuracy" in result["result"]:
                    logger.info(f"最终准确率: {result['result']['accuracy']}")
                else:
                    logger.info("流程测试完成")
        else:
            logger.error("❌ 完整流程测试失败")
            if result and "log" in result:
                logger.error(f"错误信息: {result['log']}")

        # 打印测试结果统计
        logger.info(f"\n{'=' * 50}")
        logger.info(f"🎯 测试结果统计")
        logger.info(f"{'=' * 50}")
        logger.info(f"总计测试数: {total_tests}")
        logger.info(f"通过测试数: {passed_tests}")
        logger.info(f"失败测试数: {total_tests - passed_tests}")
        logger.info(f"通过率: {passed_tests / total_tests * 100:.1f}%")

        if passed_tests == total_tests:
            logger.info("🎉 所有测试都通过了！")
        else:
            logger.warning(f"⚠️  有 {total_tests - passed_tests} 个测试失败")

        # 如果有未通过的测试，返回非零退出码
        if passed_tests < total_tests:
            sys.exit(1)

    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        sys.exit(1)

def  main1():
    """
    主函数：运行所有测试
    """
    np.random.seed(42)
    success, result = test_single_module('algorithm_knn')
    if not success:
        sys.exit(1)

def main_pipeline_only():
    """
    只测试完整流程的函数
    """
    np.random.seed(42)
    logger.info("=== 测试完整流程 ===")
    success, result = test_pipeline()
    if success:
        logger.info("✅ 完整流程测试通过")
        if result and "result" in result:
            # 修改这里：适配简化版的结果结构
            if "accuracy" in result["result"]:
                logger.info(f"最终准确率: {result['result']['accuracy']}")
            else:
                logger.info("流程测试完成")
    else:
        logger.error("❌ 完整流程测试失败")
        if result and "log" in result:
            logger.error(f"错误信息: {result['log']}")
        sys.exit(1)

if __name__ == "__main__":

    # main_pipeline_only()
    main()