# Min-Max归一化预处理器

## 简介

Min-Max归一化是一种常用的数据预处理技术，将数据缩放到指定范围内（通常是[0,1]或[-1,1]）。该算法通过线性变换将原始数据映射到新的数值范围，保持数据的相对关系不变。

## 目录结构

```
preprocessor_min-max/
├── run.py                      # 【必需】标准运行接口
├── model.py                    # 【必需】核心算法逻辑
├── config.json                 # 【必需】默认参数配置
├── meta.json                   # 【必需】元信息
├── README.md                   # 【必需】使用说明
├── requirements.txt            # 【必需】依赖列表
├── make.py                     # 【可选】数据生成脚本
└── data/                       # 【必需】示例输入输出数据
    ├── example_input.npy
    ├── example_output.npy
    └── example_labels.npy
```

## 依赖项

```bash
pip install numpy
```

## 使用方法

### 1. 生成示例数据

```bash
python make.py
```

### 2. 运行算法

#### 输入数据格式

`run.py` 接受字典类型的输入数据，格式如下：

```python
input_data = {
    "signal": np.ndarray,             # 输入信号数据（必需）
    "normalization_range": tuple,     # 归一化范围（可选，默认(0,1)）
    "axis": int/tuple,               # 归一化轴（可选，None为全局）
    "mode": str                      # 运行模式（必需）
}
```

**详细说明：**

1. **signal**: 输入信号数据
   - 数据类型：`np.ndarray`
   - 形状：任意维度，如`(batch_size, channels, signal_length)`
   - 数据类型：通常为float32或float64

2. **normalization_range**: 归一化目标范围
   - 数据类型：`tuple`
   - 默认值：`(0, 1)`
   - 常用值：`(0, 1)`, `(-1, 1)`, `(0, 10)`等

3. **axis**: 归一化轴
   - 数据类型：`int`, `tuple` 或 `None`
   - `None`: 全局归一化（默认）
   - `0`: 按第0轴归一化
   - `(1, 2)`: 按多个轴归一化

4. **mode**: 运行模式
   - `"transform"`: 变换模式 - 使用已知参数归一化
   - `"fit_transform"`: 拟合变换模式 - 计算统计参数并归一化
   - `"inverse_transform"`: 逆变换模式 - 恢复原始数据

#### 输出数据格式

```python
{
    "result": np.ndarray,        # 归一化后的信号
    "metrics": dict,             # 归一化统计信息
    "log": str,                 # 日志信息
    "success": bool             # 是否成功
}
```

**不同模式的输出内容：**

1. **fit_transform模式**：
```python
{
    "result": normalized_array,      # 归一化后的数组
    "metrics": {
        "original_min": -10.5,       # 原始最小值
        "original_max": 25.3,        # 原始最大值
        "normalized_min": 0.0,       # 归一化后最小值
        "normalized_max": 1.0,       # 归一化后最大值
        "scale_factor": 0.0279,      # 缩放因子
        "offset": 0.293              # 偏移量
    },
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
