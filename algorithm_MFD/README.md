# 匹配滤波检测算法 (MFD)

## 简介

匹配滤波检测(MFD)是一种常用的频谱感知算法，通过已知信号模板与接收信号的相关性来检测信号的存在。该算法基于7位巴克码序列，使用BPSK调制生成已知模板，通过相关运算检测目标信号，适用于主用户信号特征已知的频谱感知场景。

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
└── data/                   # 示例输入输出数据
    ├── example_input.npy
    └── example_labels.npy
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
    "labels": np.ndarray,     # 标签数据（可选，evaluate模式需要）
    "mode": str              # 运行模式（必需）
}
```

**详细说明：**

1. **signal**: I/Q信号数据
   - 数据类型：`np.ndarray`
   - 形状：`(N, signal_length, 2)` 或 `(signal_length, 2)`
   - 格式：最后一维是[I, Q]分量（实部和虚部）
   - 示例：`signal.shape = (1000, 8192, 2)` 表示1000个样本，每个8192个I/Q采样点
   - 注意：算法会自动加载已知信号模板进行匹配滤波

2. **labels**: 标签数据（可选）
   - 数据类型：`np.ndarray`
   - 形状：`(N,)`
   - 取值：0表示无目标信号，1表示有目标信号
   - 使用场景：evaluate模式必需，predict模式不需要

3. **mode**: 运行模式
   - 数据类型：`str`
   - 可选值：
     - `"predict"`: 预测模式 - 对输入信号进行匹配滤波检测
     - `"evaluate"`: 评估模式 - 需要labels，计算性能指标

#### 输出数据格式

`run.py` 返回统一格式的字典，包含以下字段：

```python
{
    "result": list,          # 主结果（检测结果数组）
    "metrics": dict,         # 评估指标（可为空字典）
    "log": str,             # 日志信息
    "success": bool         # 是否成功执行
}
```

**不同模式的输出内容：**

1. **predict模式**：
```python
{
    "result": [0, 1, 0, 1, 0, ...],            # 检测结果数组
    "metrics": {},                              # 空字典
    "log": "匹配滤波检测 - 预测模式\n输入信号形状: (1000, 8192, 2)\n检测阈值: 0.3000\n检测到信号数量: 456/1000 个\n模板类型: bpsk, 长度: 56",
    "success": True
}
```

2. **evaluate模式**：
```python
{
    "result": [0, 1, 0, 1, 0, ...],            # 检测结果数组
    "metrics": {
        "detection_rate": 0.8250,              # 检测率
        "false_alarm_rate": 0.0180,            # 虚警率  
        "accuracy": 0.8935,                    # 准确率
        "precision": 0.8654,                   # 精确率
        "f1_score": 0.8447,                    # F1分数
        "recall": 0.8250                       # 召回率
    },
    "log": "匹配滤波检测 - 评估模式\n算法性能指标:\n  检测率(召回率): 0.8250\n  虚警率: 0.0180\n  准确率: 0.8935\n  精确率: 0.8654\n  F1分数: 0.8447\n混淆矩阵: 真正例=412, 真负例=481, 假正例=9, 假负例=87\n模板类型: bpsk, 长度: 56",
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
signal_data = np.load("data/example_input.npy")  # shape: (1000, 8192, 2)
label_data = np.load("data/example_labels.npy")  # shape: (1000,)

# 预测模式
input_data = {
    "signal": signal_data,
    "mode": "predict"
}
result = run(input_data)
print(f"检测到信号: {np.sum(result['result'])}/{len(result['result'])} 个")

# 评估模式
input_data = {
    "signal": signal_data,
    "labels": label_data,
    "mode": "evaluate"
}
result = run(input_data)
print(f"检测率: {result['metrics']['detection_rate']:.3f}")
print(f"虚警率: {result['metrics']['false_alarm_rate']:.3f}")
print(f"准确率: {result['metrics']['accuracy']:.3f}")
```

## 配置参数

`config.json` 文件包含以下可配置参数：

```json
{
    "threshold": 0.3,                # 相关检测阈值
    "template_type": "bpsk",         # 模板类型
    "samples_per_bit": 8             # 每比特采样数
}
```

**参数说明：**
- **threshold**: 相关检测阈值，范围[0,1]，用于判决信号是否存在
- **template_type**: 模板信号类型，目前支持"bpsk"
- **samples_per_bit**: 每个比特的采样点数，影响模板长度

## 算法原理

匹配滤波检测(MFD)是一种基于已知信号模板的检测方法：

1. **信号模板**：使用7位巴克码 `[1, 1, 1, -1, -1, 1, -1]` 生成BPSK调制的I/Q模板
2. **模板特性**：
   - 编码序列：7位巴克码，具有良好的自相关特性
   - 调制方式：BPSK (Binary Phase Shift Keying)
   - 每符号采样：8个采样点
   - 总长度：7 × 8 = 56 个采样点
3. **匹配滤波**：计算接收信号与已知模板的相关性
4. **阈值判决**：将相关峰值与预设阈值比较进行检测判决

算法步骤：
- 加载已知信号模板（7位巴克码BPSK信号）
- 将I/Q数据转换为复数信号
- 执行匹配滤波相关运算
- 计算相关峰值并与阈值比较
- 输出检测结果

## 性能指标

算法输出以下性能指标：
- **检测率**: 正确检测到目标信号的概率
- **虚警率**: 误检测的概率
- **准确率**: 整体正确分类的概率
- **精确率**: 检测为有信号中实际有信号的比例
- **召回率**: 实际有信号中被正确检测的比例
- **F1分数**: 精确率和召回率的调和平均

## 数据格式

### 输入信号
- **格式**: I/Q数据，形状为`(N, signal_length, 2)`
- **内容**: 最后一维为[I, Q]分量（实部和虚部）
- **模板**: 算法会自动加载已知的7位巴克码模板

### 已知模板
- **类型**: 7位巴克码 `[1, 1, 1, -1, -1, 1, -1]`
- **调制**: BPSK调制
- **采样**: 每比特8个采样点
- **总长度**: 56个复数采样点
- **特点**: 具有良好的自相关特性，适合用作同步序列

## 测试数据

测试数据通过 `make.py` 生成，包含：
- **数据来源**: 自动生成的I/Q信号
- **无信号样本**: 纯高斯白噪声
- **有信号样本**: 7位巴克码模板 + 高斯白噪声
- **SNR范围**: -5dB 到 +15dB
- **模板位置**: 在信号中随机插入
- **数据格式**: `(N, 8192, 2)` - N个样本，每个8192个I/Q点