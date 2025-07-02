# Z-score标准化预处理器

## 简介

Z-score标准化（也称为标准化或Z变换）是一种常用的数据预处理技术，将数据变换为均值为0、标准差为1的标准正态分布。该算法通过减去均值并除以标准差来消除不同特征之间的量纲差异。

## 目录结构

```
preprocessor_Z-score/
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
    "signal": np.ndarray,         # 输入信号数据（必需）
    "axis": int/tuple,           # 标准化轴（可选，None为全局）
    "ddof": int,                 # 自由度修正（可选，默认0）
    "method": str,               # 标准化方法（可选）
    "mode": str                  # 运行模式（必需）
}
```

**详细说明：**

1. **signal**: 输入信号数据
   - 数据类型：`np.ndarray`
   - 形状：任意维度，如`(batch_size, channels, signal_length)`
   - 数据类型：通常为float32或float64

2. **axis**: 标准化轴
   - 数据类型：`int`, `tuple` 或 `None`
   - `None`: 全局标准化（默认）
   - `0`: 按第0轴标准化
   - `(1, 2)`: 按多个轴标准化

3. **ddof**: 自由度修正
   - 数据类型：`int`
   - 默认值：`0`（总体标准差）
   - `1`：样本标准差

4. **method**: 标准化方法
   - `"standard"`: 标准Z-score（默认）
   - `"robust"`: 鲁棒标准化（基于中位数和MAD）
   - `"quantile"`: 分位数标准化

5. **mode**: 运行模式
   - `"fit_transform"`: 拟合变换模式 - 计算统计参数并标准化
   - `"transform"`: 变换模式 - 使用已知参数标准化
   - `"inverse_transform"`: 逆变换模式 - 恢复原始数据

#### 输出数据格式

```python
{
    "result": np.ndarray,        # 标准化后的信号
    "metrics": dict,             # 标准化统计信息
    "log": str,                 # 日志信息
    "success": bool             # 是否成功
}
```

**不同模式的输出内容：**

1. **fit_transform模式**：
```python
{
    "result": standardized_array,    # 标准化后的数组
    "metrics": {
        "original_mean": 2.5,        # 原始均值
        "original_std": 1.8,         # 原始标准差
        "standardized_mean": 0.0,    # 标准化后均值
        "standardized_std": 1.0,     # 标准化后标准差
        "axis": None,                # 标准化轴
        "ddof": 0                    # 自由度修正
    },
    "log": "Z-score标准化 - fit_transform模式\n...",
    "success": True
}
```

2. **inverse_transform模式**：
```python
{
    "result": recovered_array,       # 恢复的原始数组
    "metrics": {
        "original_mean": 2.5,
        "original_std": 1.8,
        "recovered_mean": 2.499,
        "recovered_std": 1.801
    },
    "log": "Z-score标准化 - inverse_transform模式\n...",
    "success": True
}
```

#### 使用示例

```python
import numpy as np
from run import run

# 加载测试数据
signal_data = np.load("data/example_input.npy")  # shape: (100, 2, 1000)

# 1. 标准Z-score标准化
input_data = {
    "signal": signal_data,
    "mode": "fit_transform"
}
result = run(input_data)
print(f"标准化后统计: 均值={result['result'].mean():.6f}, 标准差={result['result'].std():.6f}")

# 2. 鲁棒标准化
input_data = {
    "signal": signal_data,
    "method": "robust",
    "center_method": "median",
    "scale_method": "mad",
    "mode": "fit_transform"
}
result = run(input_data)

# 3. 按轴标准化
input_data = {
    "signal": signal_data,
    "axis": -1,  # 按最后一个轴标准化
    "mode": "fit_transform"
}
result = run(input_data)

# 4. 逆变换恢复原始数据
input_data = {
    "signal": result['result'],
    "original_mean": result['metrics']['original_mean'],
    "original_std": result['metrics']['original_std'],
    "mode": "inverse_transform"
}
recovered = run(input_data)
```

### 3. 配置参数

可以通过修改`config.json`文件调整算法参数：

```json
{
    "ddof": 0,                       # 自由度修正
    "clip_outliers": false,          # 是否裁剪异常值
    "outlier_threshold": 3.0,        # 异常值阈值（标准差倍数）
    "robust_method": false,          # 是否使用鲁棒方法
    "center_method": "mean",         # 中心化方法（mean/median）
    "scale_method": "std",           # 缩放方法（std/mad/iqr）
    "feature_selection": {
        "use_global_norm": true,     # 使用全局标准化
        "use_channel_norm": false,   # 使用通道标准化
        "use_robust_norm": false,    # 使用鲁棒标准化
        "use_quantile_norm": false   # 使用分位数标准化
    }
}
```

## 算法原理

Z-score标准化使用以下公式：

```
Z = (X - μ) / σ
```

其中：
- `X`: 原始数据
- `μ`: 数据均值
- `σ`: 数据标准差
- `Z`: 标准化后的数据

### 支持的标准化方法

1. **标准Z-score标准化**：
   - 公式：`Z = (X - mean) / std`
   - 特点：结果服从标准正态分布N(0,1)

2. **鲁棒标准化**：
   - 基于中位数：`Z = (X - median) / MAD`
   - MAD：绝对中位差（Median Absolute Deviation）
   - 特点：对异常值更稳健

3. **分位数标准化**：
   - 基于四分位距：`Z = (X - median) / IQR`
   - IQR：四分位距（Interquartile Range）
   - 特点：使用分位数，不受极值影响

4. **裁剪标准化**：
   - 先标准化，再裁剪到±3σ范围
   - 特点：限制异常值的影响

### 数学特性

- **均值**: 标准化后均值为0
- **方差**: 标准化后方差为1
- **分布形状**: 保持原始分布的形状
- **可逆性**: 完全可逆，能精确恢复原始数据

## 性能特点

- **时间复杂度**: O(n)，其中n为数据点数量
- **空间复杂度**: O(n)
- **实时性**: 支持实时处理
- **数值稳定性**: 处理了零方差等边界情况
- **可扩展性**: 支持多维数据和按轴处理

## 应用场景

1. **机器学习预处理**: 统一特征尺度，加速收敛
2. **信号处理**: 消除直流分量，标准化幅度
3. **统计分析**: 标准化数据用于比较和分析
4. **神经网络**: 改善梯度流动，提高训练稳定性
5. **异常检测**: 基于标准化后的Z值检测异常

## 与Min-Max归一化的区别

| 特性 | Z-score标准化 | Min-Max归一化 |
|------|---------------|---------------|
| 输出范围 | 不固定（通常-3到+3） | 固定范围（如[0,1]） |
| 分布形状 | 保持原始分布 | 保持原始分布 |
| 异常值敏感性 | 较低 | 很高 |
| 适用场景 | 正态分布数据 | 已知数据范围 |
| 可解释性 | Z值表示偏离程度 | 相对位置 |

## 注意事项

1. Z-score标准化假设数据近似正态分布
2. 对于严重偏态或多峰分布，建议使用鲁棒方法
3. 当标准差为0时，算法会返回零数组避免除零错误
4. 对于测试数据，应使用训练数据的均值和标准差
5. 逆变换需要保存原始的均值和标准差参数
6. 异常值可能影响标准化效果，可考虑先处理异常值
