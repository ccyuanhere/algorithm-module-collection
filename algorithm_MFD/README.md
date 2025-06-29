# 匹配滤波检测 (Matched Filter Detection)

## 算法简介

匹配滤波检测是一种最优的信号检测算法，通过设计与已知信号波形匹配的滤波器来实现最大信噪比增益。该算法特别适用于已知发送信号特征的通信系统和频谱感知场景。

### 核心原理

1. **模板匹配**: 利用已知的信号波形作为模板
2. **最优滤波**: 滤波器冲激响应为模板信号的共轭时间反转
3. **相关检测**: 通过计算接收信号与模板的相关性来检测信号存在
4. **阈值判决**: 使用固定或自适应阈值进行最终判决

### 算法特点

- ✅ **最优性能**: 在已知信号条件下提供最佳检测性能
- ✅ **高精度**: 特别是在高信噪比环境下
- ✅ **时延估计**: 能够同时估计信号的到达时间
- ✅ **实时处理**: 计算复杂度相对较低
- ❌ **需要先验知识**: 必须事先知道信号的精确波形
- ❌ **环境敏感**: 在信号变化或多径环境下性能下降

## 使用说明

### 基本用法

```python
from run import run
import numpy as np

# 准备输入数据
input_data = {
    "signal": signal_array,  # 复数信号数组
    "labels": labels_array,  # 可选：真实标签
    "mode": "predict"        # 运行模式
}

# 执行检测
result = run(input_data)
```

### 输入数据格式

#### 方式1: 直接信号输入
```python
{
    "signal": np.ndarray,     # 复数信号，形状为 (N,) 或 (batch_size, N)
    "labels": np.ndarray,     # 可选：标签数组 (0=无信号, 1=有信号)
    "mode": "predict"         # 运行模式："predict", "evaluate"
}
```

#### 方式2: 结构化数据输入
```python
{
    "signal": {
        "signals": np.ndarray,           # 多个信号
        "known_sequence": [1,0,1,0,...], # 已知比特序列
        "fc": 1000,                      # 载波频率
        "fs": 8000,                      # 采样频率
        "duration_per_bit": 0.001        # 比特持续时间
    },
    "labels": np.ndarray,
    "mode": "evaluate"
}
```

### 输出结果格式

```python
{
    "result": {
        "results": [                     # 每个信号的检测结果
            {
                "signal_id": 0,
                "detection": True,       # 检测结果
                "peak_value": 0.85,     # 峰值
                "threshold": 0.5,       # 检测阈值
                "snr_gain": 12.3,       # 信噪比增益
                "time_delay": 0.002,    # 时延估计
                ...
            }
        ],
        "detection_summary": {           # 整体检测统计
            "total_signals": 5,
            "detections": 3,
            "detection_rate": 0.6
        },
        "metrics": {                     # 性能指标（有标签时）
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.90,
            "detection_rate": 0.89,
            "false_alarm_rate": 0.05
        }
    },
    "success": True,
    "log": "检测完成信息"
}
```

## 配置参数

### config.json 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `known_sequence` | array | [1,0,1,0,1,1,0,1] | 已知比特序列 |
| `fc` | number | 1000 | 载波频率 (Hz) |
| `fs` | number | 8000 | 采样频率 (Hz) |
| `duration_per_bit` | number | 0.001 | 每比特持续时间 (s) |
| `detection_threshold` | number/null | null | 检测阈值，null表示自适应 |
| `false_alarm_rate` | number | 0.001 | 期望虚警率 |
| `correlation_mode` | string | "full" | 相关模式："full", "valid", "same" |

### 示例配置

```json
{
    "known_sequence": [1, 0, 1, 0, 1, 1, 0, 1],
    "fc": 2000,
    "fs": 16000,
    "duration_per_bit": 0.0005,
    "detection_threshold": null,
    "false_alarm_rate": 0.01,
    "correlation_mode": "full"
}
```

## 数据生成

### 生成测试数据

```bash
python make.py
```

生成的测试数据包括：
- 不同信噪比的BPSK调制信号
- 纯噪声场景
- 干扰信号场景
- 长信号中的目标信号检测

### 数据文件

- `data/example_input.npy`: 输入信号数据
- `data/example_output.npy`: 匹配滤波器输出
- `data/example_labels.npy`: 检测标签
- `data/data_description.txt`: 数据描述文件

## 性能特征

### 适用场景

✅ **推荐使用**:
- 已知信号格式的通信系统
- 数字调制信号检测
- 同步序列检测
- 高精度要求的场景

❌ **不推荐使用**:
- 未知信号检测
- 频繁变化的信号环境  
- 强多径衰落环境
- 实时性要求极高的场景

### 性能指标

| 指标 | 典型值 | 说明 |
|------|--------|------|
| 检测概率 | >0.9 (SNR>0dB) | 高SNR下检测率很高 |
| 虚警概率 | <0.01 | 可通过阈值控制 |
| 时延精度 | ±1个采样点 | 依赖采样率 |
| 计算复杂度 | O(N×M) | N信号长度，M模板长度 |

## 算法原理

### 数学基础

对于接收信号 r(t)，匹配滤波器的冲激响应为:
```
h(t) = s*(T-t)
```
其中 s(t) 是已知信号，* 表示共轭，T 是信号持续时间。

滤波器输出:
```
y(t) = ∫ r(τ)h(t-τ)dτ = ∫ r(τ)s*(T-t+τ)dτ
```

### 检测判决

```
H1: y(t) > threshold  (信号存在)
H0: y(t) ≤ threshold  (无信号)
```

## 依赖项

详见 `requirements.txt`：
- numpy
- scipy (可选，用于高级信号处理)

## 版本历史

- v2.0: 完整的匹配滤波检测实现，支持BPSK信号
- v1.0: 基础版本
