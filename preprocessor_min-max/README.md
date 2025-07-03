# Min-Max归一化预处理器

## 🎯 算法概述

Min-Max归一化是一种常用的数据预处理技术，将数据线性映射到指定范围内（如[0,1]或[-1,1]）。该算法通过线性变换保持数据的相对关系不变，是机器学习和信号处理中的基础预处理方法。

**核心公式：**
```
X_norm = (X - X_min) / (X_max - X_min) * (target_max - target_min) + target_min
```

## 📁 目录结构

```
preprocessor_min-max/
├── run.py                      # 【必需】标准API接口
├── model.py                    # 【必需】核心算法逻辑
├── config.json                 # 【必需】默认参数配置
├── meta.json                   # 【必需】算法元信息
├── README.md                   # 【必需】使用说明
├── requirements.txt            # 【必需】依赖列表
├── make.py                     # 【可选】测试数据生成脚本
└── data/                       # 【必需】示例数据
    └── example_input.npy       # 输入信号样本
```

## 🚀 快速开始

### 1. 生成测试数据

```bash
python make.py
```

### 2. 运行算法测试

```bash
python run.py
```

将展示详细的归一化效果，包括：
- 原始信号统计信息
- 不同目标范围的归一化结果对比
- 逆变换精度验证
- 生成可视化对比图片

## 📚 API接口

### 标准调用接口

```python
from run import run

# 输入数据结构
input_data = {
    "signal": np.ndarray,             # 输入信号数据（必需）
    "normalization_range": tuple,     # 归一化范围（可选，默认(0,1)）
    "axis": int/tuple,               # 归一化轴（可选，None为全局）
    "mode": str                      # 运行模式（必需）
}

# 调用算法
result = run(input_data)
```

### 输入参数详解

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `signal` | `np.ndarray` | 输入信号数据，支持多维数组 | - |
| `normalization_range` | `tuple` | 目标范围，如`(0,1)`, `(-1,1)` | `(0,1)` |
| `axis` | `int/tuple/None` | 归一化轴，None表示全局归一化 | `None` |
| `mode` | `str` | 运行模式：`fit_transform`、`inverse_transform` | - |

### 输出结果格式

```python
{
    "result": np.ndarray,        # 归一化后的信号
    "metrics": dict,             # 统计信息和变换参数
    "log": str,                 # 详细日志信息
    "success": bool             # 执行状态
}
```

## 🔧 使用示例

### 基本归一化

```python
import numpy as np
from run import run

# 加载测试数据
signal_data = np.load("data/example_input.npy")  # shape: (100, 2, 1000)

# 归一化到[0,1]
input_data = {
    "signal": signal_data,
    "normalization_range": (0, 1),
    "mode": "fit_transform"
}
result = run(input_data)

print(f"原始范围: [{signal_data.min():.4f}, {signal_data.max():.4f}]")
print(f"归一化后: [{result['result'].min():.4f}, {result['result'].max():.4f}]")
```

### 不同目标范围

```python
# 归一化到[-1,1]
input_data = {
    "signal": signal_data,
    "normalization_range": (-1, 1),
    "mode": "fit_transform"
}
result = run(input_data)

# 归一化到[0,10]
input_data = {
    "signal": signal_data,
    "normalization_range": (0, 10),
    "mode": "fit_transform"
}
result = run(input_data)
```

### 逆变换恢复

```python
# 先归一化
input_data = {
    "signal": signal_data,
    "normalization_range": (0, 1),
    "mode": "fit_transform"
}
normalized_result = run(input_data)

# 逆变换恢复原始数据
input_data = {
    "signal": normalized_result['result'],
    "original_min": normalized_result['metrics']['original_min'],
    "original_max": normalized_result['metrics']['original_max'],
    "mode": "inverse_transform"
}
recovered_result = run(input_data)

# 验证恢复精度
recovery_error = np.mean(np.abs(signal_data - recovered_result['result']))
print(f"恢复误差: {recovery_error:.8f}")
```

## 📊 归一化效果展示

### 输入信号特征
- **多样化信号类型**: 正弦波、线性调频、噪声、混合信号
- **多种幅度范围**: 从微小值(±0.001)到大值(±10000)
- **多通道支持**: 支持I/Q双通道或多通道信号

### 归一化效果

| 原始范围 | 目标范围 | 归一化后范围 | 精度 |
|----------|----------|--------------|------|
| [-2543.2, 8712.5] | [0, 1] | [0.0000, 1.0000] | 完美 |
| [-0.0008, 0.0012] | [-1, 1] | [-1.0000, 1.0000] | 完美 |
| [1205.3, 9876.1] | [0, 10] | [0.0000, 10.0000] | 完美 |

### 变换参数

算法自动计算并返回变换参数：
- **original_min/max**: 原始数据的最小/最大值
- **scale_factor**: 缩放因子 = (target_max - target_min) / (original_max - original_min)
- **offset**: 偏移量 = target_min - original_min * scale_factor

## ⚙️ 配置参数

通过修改`config.json`自定义算法行为：

```json
{
    "default_range": [0, 1],         # 默认归一化范围
    "clip_outliers": false,          # 是否裁剪异常值
    "preserve_zero": false           # 是否保持零值不变
}
```

### 参数说明

- **default_range**: 默认的归一化目标范围，当用户未指定时使用
- **clip_outliers**: 是否在归一化前裁剪异常值，可提高鲁棒性  
- **preserve_zero**: 是否在归一化过程中保持零值的特殊位置
```

## 🎨 算法特点

### ✅ 优势
- **线性变换**: 保持数据相对关系不变
- **完全可逆**: 支持精确的逆变换
- **实时处理**: 计算复杂度O(n)，适合实时应用
- **多维支持**: 处理任意维度的信号数据
- **数值稳定**: 处理边界情况（如min=max）

### ⚠️ 注意事项
- 对异常值敏感，极值会影响整体缩放
- 新数据应使用训练数据的归一化参数
- 数据分布变化时需重新计算参数

## 📈 应用场景

1. **机器学习预处理**: 统一特征量级，加速模型收敛
2. **信号处理**: 将信号幅度标准化到ADC范围
3. **神经网络**: 激活函数输入范围优化
4. **数据可视化**: 映射数据到显示坐标系
5. **通信系统**: 信号功率归一化

## 🔍 技术细节

### 算法复杂度
- **时间复杂度**: O(n)，其中n为数据点数量
- **空间复杂度**: O(n)，需要存储输出数据
- **内存占用**: 约为输入数据的1-2倍

### 数值精度
- **浮点运算**: 使用双精度浮点数计算
- **逆变换精度**: 典型误差 < 1e-10
- **边界处理**: 自动处理min=max的退化情况

## 📦 依赖项

```txt
numpy>=1.19.0
```

## 🛠️ 开发说明

### 核心文件
- `model.py`: 实现具体的归一化算法
- `run.py`: 提供标准化API接口
- `config.json`: 算法参数配置
- `meta.json`: 算法元信息描述

### 测试数据
- `make.py`: 生成多样化的测试信号
- `data/example_input.npy`: 包含不同类型、不同范围的测试信号

### 可视化输出
运行`python run.py`会在`assets/`目录生成归一化效果对比图：
- `归一化对比_0_1.png`: [0,1]范围归一化对比
- `归一化对比_neg1_1.png`: [-1,1]范围归一化对比  
- `归一化对比_0_10.png`: [0,10]范围归一化对比
- `归一化对比_neg5_5.png`: [-5,5]范围归一化对比

运行`python run.py`查看详细的归一化效果演示！
    "log": "Min-Max归一化 - fit_transform模式\n...",
    "success": True
}
```

2. **inverse_transform模式**：
```python
{
    "result": recovered_array,       # 恢复的原始数组
    "metrics": {
        "original_min": -10.5,
        "original_max": 25.3,
        "recovered_min": -10.499,
        "recovered_max": 25.301
    },
    "log": "Min-Max归一化 - inverse_transform模式\n...",
    "success": True
}
```

#### 使用示例

```python
import numpy as np
from run import run

# 加载测试数据
signal_data = np.load("data/example_input.npy")  # shape: (100, 2, 1000)

# 1. 标准归一化到[0,1]
input_data = {
    "signal": signal_data,
    "normalization_range": (0, 1),
    "mode": "fit_transform"
}
result = run(input_data)
print(f"归一化范围: [{result['result'].min():.4f}, {result['result'].max():.4f}]")

# 2. 归一化到[-1,1]
input_data = {
    "signal": signal_data,
    "normalization_range": (-1, 1),
    "mode": "fit_transform"
}
result = run(input_data)

# 3. 逆变换恢复原始数据
input_data = {
    "signal": result['result'],
    "original_min": result['metrics']['original_min'],
    "original_max": result['metrics']['original_max'],
    "mode": "inverse_transform"
}
recovered = run(input_data)
```

### 3. 配置参数

可以通过修改`config.json`文件调整算法参数：

```json
{
    "default_range": [0, 1],         # 默认归一化范围
    "clip_outliers": false,          # 是否裁剪异常值
    "outlier_percentile": 1.0,       # 异常值百分位阈值
    "preserve_zero": false,          # 是否保持零值不变
    "robust_quantiles": [0.25, 0.75], # 鲁棒归一化分位数
    "feature_selection": {
        "use_global_norm": true,     # 使用全局归一化
        "use_channel_norm": false,   # 使用通道归一化
        "use_robust_norm": false     # 使用鲁棒归一化
    }
}
```

## 算法原理

Min-Max归一化使用以下公式：

```
X_norm = (X - X_min) / (X_max - X_min) * (range_max - range_min) + range_min
```

其中：
- `X`: 原始数据
- `X_min`, `X_max`: 原始数据的最小值和最大值
- `range_min`, `range_max`: 目标范围的最小值和最大值

### 支持的归一化方法

1. **标准Min-Max归一化**：基于全局最小值和最大值
2. **鲁棒归一化**：基于分位数，对异常值更稳健
3. **保持零值归一化**：确保零值在归一化后仍为特定值
4. **按轴归一化**：可以按指定轴进行归一化

## 性能特点

- **时间复杂度**: O(n)，其中n为数据点数量
- **空间复杂度**: O(n)
- **实时性**: 支持实时处理
- **可逆性**: 支持完全可逆的逆变换
- **数值稳定性**: 处理了除零等边界情况

## 应用场景

1. **机器学习预处理**: 为算法提供标准化输入
2. **信号处理**: 将信号幅度标准化到特定范围
3. **数据可视化**: 将数据映射到显示范围
4. **特征工程**: 统一不同特征的数值范围
5. **神经网络**: 加速收敛，提高训练稳定性

## 注意事项

1. Min-Max归一化对异常值敏感，极值会影响整体缩放
2. 当数据分布发生变化时，需要重新计算归一化参数
3. 对于测试数据，应使用训练数据的归一化参数
4. 如果最大值等于最小值，算法会返回中间值避免除零错误
5. 逆变换需要保存原始的最小值和最大值参数
