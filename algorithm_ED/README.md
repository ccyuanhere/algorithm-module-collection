# 能量检测频谱感知算法

## 算法简介
本算法实现了基于能量检测的频谱感知功能，通过计算接收信号的能量并与预设阈值比较来判断主用户是否存在。能量检测是认知无线电系统中最常用的频谱感知方法之一，具有实现简单、计算复杂度低的特点。算法采用传统的噪声底阈值计算方法，支持预测和评估两种运行模式。

## 输入输出说明

### 输入
输入为字典格式，必须包含以下字段：
```python
{
    "signal": np.ndarray,  # 输入I/Q信号数据，形状为 (batch_size, signal_length, 2)
}
```

可选字段：
```python
{
    "labels": np.ndarray,  # 标签数据，形状为 (batch_size,)，用于评估模式
    "mode": str           # 运行模式，可选 "predict" 或 "evaluate"，默认为 "predict"
}
```

**详细说明：**

1. **signal**: I/Q信号数据
   - 数据类型：`np.ndarray`
   - 形状：`(batch_size, signal_length, 2)`
   - 格式：最后一维是[I, Q]分量（实部和虚部）
   - 示例：`signal.shape = (100, 8192, 2)` 表示100个样本，每个8192个I/Q采样点

2. **labels**: 标签数据（仅评估模式需要）
   - 数据类型：`np.ndarray`
   - 形状：`(batch_size,)`
   - 取值：0表示无主用户信号，1表示有主用户信号
   - 使用场景：evaluate模式必需，predict模式不需要

3. **mode**: 运行模式
   - 数据类型：`str`
   - 可选值：
     - `"predict"`: 预测模式 - 对输入信号进行检测，返回检测结果
     - `"evaluate"`: 评估模式 - 需要labels，计算性能指标

### 输出
根据运行模式返回不同的结果，统一格式为：
```python
{
    "result": np.ndarray,    # 主结果（检测结果数组）
    "metrics": dict,         # 评估指标（可为空字典）
    "log": str,             # 日志信息
    "success": bool         # 是否成功执行
}
```

**不同模式的输出内容：**

1. **预测模式（mode="predict"）**：
```python
{
    "result": np.array([0, 1, 0, 1, ...]),  # 检测结果：0=无信号，1=有信号
    "metrics": {},                          # 空字典
    "log": "能量检测算法 - 预测模式\n输入信号形状: (100, 8192, 2)\n计算窗口化能量: 窗口大小=1024, 窗口数量=8\n阈值计算方法: noise_floor, 阈值=1234.5678\n完成能量检测，检测到信号的样本数: 45/100\n",
    "success": True
}
```

2. **评估模式（mode="evaluate"）**：
```python
{
    "result": np.array([0, 1, 0, 1, ...]),  # 检测结果
    "metrics": {
        "accuracy": 0.85,                   # 准确率
        "detection_rate": 0.88,             # 检测率（真正率）
        "false_alarm_rate": 0.12,           # 虚警率
        "recall": 0.88,                     # 召回率（同检测率）
        "precision": 0.82,                  # 精确率
        "tp": 44,                          # 真正例数量
        "tn": 41,                          # 真负例数量
        "fp": 6,                           # 假正例数量
        "fn": 9                            # 假负例数量
    },
    "log": "能量检测算法 - 评估模式\n输入信号形状: (100, 8192, 2)\n...\n评估结果: 检测率=0.880, 虚警率=0.128, 准确率=0.850, 召回率=0.880\n",
    "success": True
}
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成示例数据
首先需要生成示例数据用于测试：
```bash
python make.py
```
这将在`data/`目录下生成示例输入和标签数据。

### 3. 基本调用

#### 预测模式
```python
from algorithm_ED.run import run
import numpy as np

# 加载示例数据
signal = np.load("data/example_input.npy")

# 预测模式
result = run({
    "signal": signal,
    "mode": "predict"
})

print(f"检测结果: {result['result']}")
print(f"日志信息: {result['log']}")
```

#### 评估模式
```python
# 加载数据和标签
signal = np.load("data/example_input.npy")
labels = np.load("data/example_labels.npy")

# 评估模式
result = run({
    "signal": signal,
    "labels": labels,
    "mode": "evaluate"
})

print(f"检测结果: {result['result']}")
print(f"性能指标: {result['metrics']}")
print(f"检测率: {result['metrics']['detection_rate']:.3f}")
print(f"虚警率: {result['metrics']['false_alarm_rate']:.3f}")
```

### 4. 使用自定义配置
```python
result = run({
    "signal": signal,
    "mode": "predict"
}, config_path="path/to/your/config.json")
```

### 5. 作为模块导入
```python
from algorithm_ED.run import run

# 准备您的信号数据
signal = your_signal_data  # 形状: (batch_size, signal_length, 2)

# 运行算法
result = run({
    "signal": signal,
    "mode": "predict"
})
```

## 文件说明
```
algorithm_ED/
├── run.py                      # 标准运行接口
├── model.py                    # 核心算法逻辑
├── config.json                 # 默认参数配置
├── meta.json                   # 元信息
├── README.md                   # 使用说明
├── requirements.txt            # 依赖列表
├── make.py                     # 数据生成脚本
└── data/                       # 示例数据目录
    ├── example_input.npy       # 示例输入信号
    └── example_labels.npy      # 示例标签数据
```

## 配置参数说明
配置文件 `config.json` 包含以下参数：

```json
{
    "window_size": 1024,
    "threshold_factor": 2.0,
    "noise_estimation_method": "percentile"
}
```

**参数详细说明：**

- **window_size** (默认: 1024)
  - 类型: `int`
  - 说明: 能量检测窗口大小（采样点数）
  - 作用: 决定每个窗口包含多少个信号样本进行能量计算
  - 取值范围: 1 到 signal_length
  - 影响: 窗口越大，计算越平滑但时间分辨率越低

- **threshold_factor** (默认: 2.0)
  - 类型: `float`
  - 说明: 阈值调整因子
  - 作用: 与噪声基准相乘得到最终检测阈值
  - 取值范围: > 1.0（推荐1.5-5.0）
  - 影响: 数值越大，阈值越高，检测越保守，虚警率越低但检测率也可能降低

- **noise_estimation_method** (默认: "percentile")
  - 类型: `str`
  - 说明: 噪声基准估计方法
  - 可选值:
    - `"minimum"`: 使用能量分布的直方图左峰作为噪声基准
    - `"percentile"`: 使用20%分位数作为噪声基准（更鲁棒）
  - 影响: percentile方法通常更稳定，适合大多数场景

## 算法原理

### 核心思想
能量检测算法基于一个简单的假设：主用户信号的存在会显著增加接收信号的能量水平。算法通过比较观测能量与噪声基准来做出检测决策。

### 算法流程
1. **信号预处理**: 将I/Q信号转换为复数形式
2. **窗口化处理**: 将信号分割成固定大小的时间窗口
3. **能量计算**: 计算每个窗口的能量 E = Σ|x[n]|²
4. **噪声基准估计**: 使用统计方法估计噪声水平
5. **阈值计算**: threshold = noise_floor × threshold_factor
6. **检测判决**: 如果任一窗口能量超过阈值，判定为有信号

### 数学表达式
- **窗口能量**: E_i = Σ_{n=i×L}^{(i+1)×L-1} |x[n]|²
- **噪声基准**: N₀ = percentile(E, 20%) 或 histogram_peak(E)
- **检测阈值**: T = N₀ × threshold_factor
- **检测结果**: D = 1 if max(E_i) > T else 0

### 优势与限制
**优势:**
- 实现简单，计算复杂度低
- 不需要先验知识或信号模板
- 对各种调制方式都有效
- 适用于宽带频谱感知

**限制:**
- 对信噪比要求较高
- 容易受强噪声干扰影响
- 无法区分不同类型的信号

## 性能评估指标

算法在评估模式下计算以下指标：

- **准确率 (Accuracy)**: (TP + TN) / (TP + TN + FP + FN)
  - 正确检测的样本占总样本的比例

- **检测率 (Detection Rate)**: TP / (TP + FN)
  - 有信号时正确检测的概率，越高越好

- **虚警率 (False Alarm Rate)**: FP / (FP + TN)
  - 无信号时误报的概率，越低越好

- **召回率 (Recall)**: 同检测率

- **精确率 (Precision)**: TP / (TP + FP)
  - 预测为有信号的样本中真正有信号的比例

其中：
- TP: 真正例（有信号且检测到）
- TN: 真负例（无信号且未检测到）
- FP: 假正例（无信号但检测到）
- FN: 假负例（有信号但未检测到）

## 使用建议

### 参数调优
1. **window_size**: 
   - 信号变化快：选择较小窗口（512-1024）
   - 信号变化慢：可选择较大窗口（2048-4096）

2. **threshold_factor**:
   - 高SNR环境：可适当降低（1.5-2.5）
   - 低SNR环境：需要提高（2.5-4.0）
   - 要求低虚警：增大因子
   - 要求高检测率：减小因子

3. **noise_estimation_method**:
   - 一般情况：推荐使用"percentile"
   - 噪声环境复杂：尝试"minimum"

### 典型应用场景
- 认知无线电频谱感知
- 信号存在性检测
- 通信系统载波检测
- 雷达目标检测

## 依赖项
```
numpy>=1.21.0
```

## 注意事项
1. 确保输入信号维度正确：(batch_size, signal_length, 2)
2. 评估模式需要提供正确格式的标签数据
3. 窗口大小不能超过信号长度
4. 阈值因子建议在1.5-5.0范围内调整
5. 算法对强脉冲干扰敏感，可通过调整参数优化

## 更新日志
- v1.0: 初始版本
  - 实现基本的能量检测算法
  - 支持预测和评估两种模式
  - 提供噪声底阈值计算方法
  - 包含完整的性能评估指标
  - 添加示例数据生成功能