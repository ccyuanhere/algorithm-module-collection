# 匹配滤波检测算法 (MFD)

## 简介

匹配滤波检测(MFD)是一种常用的频谱感知算法，通过已知信号模板与接收信号的相关性来检测信号的存在。改进版本引入随机阈值机制和类信号噪声生成，支持通过机器学习模型优化检测决策，在信噪比适中的情况下可灵活调整检测概率与虚警概率平衡，适用于主用户信号已知的场景。

## 目录结构

```
algorithm_MFD/
├── run.py                  # 标准运行接口
├── model.py                # 核心算法逻辑
├── config.json             # 默认参数配置
├── meta.json               # 元信息
├── README.md               # 使用说明
├── requirements.txt        # 依赖列表
├── make.py                 # 数据生成脚本
├── assets/                 # 静态演示图
│   ├── algorithm_structure.png
│   └── example_visualization.png
├── data/                   # 示例输入输出数据
│   ├── example_input.npy
│   ├── example_output.npy
│   └── example_labels.npy
└── models/                 # 模型文件夹
    └── rf_model.pkl        # 随机森林模型
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成示例数据

```bash
python make.py
```

### 3. 运行算法

#### 输入数据格式

`run.py` 接受字典类型的输入数据，格式如下：

```python
input_data = {
    "signal": np.ndarray,     # 输入I/Q信号数据（必需）
    "labels": np.ndarray,     # 标签数据（可选，evaluate/train模式需要）
    "mode": str              # 运行模式（必需）
}
```

**详细说明：**

1. **signal**: I/Q信号数据
   - 数据类型：`np.ndarray`
   - 形状：`(N, signal_length, 2)` 或 `(signal_length, 2)`
   - 格式：最后一维是[I, Q]分量（实部和虚部）
   - 示例：`signal.shape = (1000, 512, 2)` 表示1000个样本，每个512个I/Q采样点
   - 注意：算法会自动加载已知信号模板进行匹配滤波

2. **labels**: 标签数据（可选）
   - 数据类型：`np.ndarray`
   - 形状：`(N,)`
   - 取值：0表示无目标信号，1表示有目标信号
   - 使用场景：evaluate和train模式必需，predict模式可选

3. **mode**: 运行模式
   - 数据类型：`str`
   - 可选值：
     - `"predict"`: 预测模式 - 对输入信号进行匹配滤波检测
     - `"evaluate"`: 评估模式 - 需要labels，计算性能指标
     - `"train"`: 训练模式 - 进行阈值校准（传统算法的参数优化）

#### 输出数据格式

`run.py` 返回统一格式的字典，包含以下字段：

```python
{
    "result": dict,          # 主结果（必需）
    "metrics": dict,         # 评估指标（可为空字典）
    "log": str,             # 日志信息
    "success": bool         # 是否成功执行
}
```

**不同模式的输出内容：**

1. **predict模式**：
```python
{
    "result": {
        "detections": [0, 1, 0, ...],           # 检测结果数组
        "detection_values": [0.25, 0.78, ...], # 相关峰值
        "detection_positions": [45, 123, ...],  # 检测位置
        "correlation_threshold": 0.3,           # 使用的阈值
        "template_info": {
            "type": "BPSK_8bit_barker",
            "length": 64
        }
    },
    "metrics": {},                              # 空字典
    "log": "匹配滤波检测 - 预测模式\n...",
    "success": True
}
```

2. **evaluate模式**：
```python
{
    "result": {
        "detections": [0, 1, 0, ...],
        "detection_values": [0.25, 0.78, ...],
        "detection_positions": [45, 123, ...],
        "correlation_threshold": 0.3,
        "confusion_matrix": {"tp": 334, "tn": 488, "fp": 3, "fn": 175}
    },
    "metrics": {
        "detection_rate": 0.656,                # 检测率
        "false_alarm_rate": 0.006,              # 虚警率  
        "accuracy": 0.822,                      # 准确率
        "recall": 0.656                         # 召回率
    },
    "log": "匹配滤波检测 - 评估模式\n...",
    "success": True
}
```

3. **train模式**：
```python
{
    "result": {
        "detections": [0, 1, 0, ...],
        "calibrated_threshold": 0.309,          # 校准后的最优阈值
        "correlation_threshold": 0.3,           # 原始阈值
        "template_info": {...}
    },
    "metrics": {
        "detection_rate": 0.945,                # 校准后检测率
        "false_alarm_rate": 0.120               # 校准后虚警率
    },
    "log": "匹配滤波检测 - 训练模式\n...",
    "success": True
}
```

#### 使用示例

```bash
python run.py
```

#### 代码调用示例

```python
import numpy as np
from run import run

# 加载示例数据
signal_data = np.load("data/example_input.npy")  # shape: (1000, 512, 2)
label_data = np.load("data/example_labels.npy")  # shape: (1000,)

# 预测模式
input_data = {
    "signal": signal_data,
    "mode": "predict"
}
result = run(input_data)
print(f"检测到信号: {np.sum(result['result']['detections'])}/{len(result['result']['detections'])} 个")

# 评估模式
input_data = {
    "signal": signal_data,
    "labels": label_data,
    "mode": "evaluate"
}
result = run(input_data)
print(f"检测率: {result['metrics']['detection_rate']:.3f}")
print(f"虚警率: {result['metrics']['false_alarm_rate']:.3f}")
```

## 配置参数

`config.json` 文件包含以下可配置参数：

```json
{
    "threshold": 0.3,                # 相关检测阈值
    "template_type": "bpsk",         # 模板类型
    "samples_per_bit": 8,            # 每比特采样数
    "carrier_freq": 2000,            # 载波频率(Hz)
    "sampling_freq": 8000,           # 采样频率(Hz)
    "num_samples": 1000,             # 样本数量
    "signal_length": 512,            # 信号长度
    "snr_range": [-2, 15],          # 信噪比范围(dB)
    "false_alarm_rate": 0.1          # 期望虚警率
}
```

## 算法原理

匹配滤波检测(MFD)是一种基于已知信号模板的检测方法：

1. **信号模板**：使用8位巴克码(Barker Code)生成BPSK调制的I/Q模板
2. **匹配滤波**：计算接收信号与已知模板的相关性
3. **相关峰值检测**：寻找相关输出的峰值位置和幅度
4. **阈值判决**：将相关峰值与预设阈值比较进行检测判决

算法步骤：
- 加载已知信号模板（8位巴克码BPSK信号）
- 将I/Q数据转换为复数信号
- 执行匹配滤波相关运算
- 计算相关峰值并与阈值比较
- 输出检测结果和位置信息

## 性能指标

算法输出以下性能指标：
- **检测率**: 正确检测到目标信号的概率
- **虚警率**: 误检测的概率
- **准确率**: 整体正确分类的概率
- **召回率**: 实际有信号中被正确检测的比例

## 数据格式

### 输入信号
- **格式**: I/Q数据，形状为`(N, signal_length, 2)`
- **内容**: 最后一维为[I, Q]分量（实部和虚部）
- **模板**: 算法会自动加载已知的8位巴克码模板

### 已知模板
- **类型**: 8位巴克码 `[1, 0, 1, 1, 0, 0, 1, 0]`
- **调制**: BPSK调制
- **采样**: 每比特8个采样点
- **存储**: 保存在`data/known_template.npy`文件中