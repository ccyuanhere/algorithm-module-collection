# FFT变换预处理器

## 简介

快速傅里叶变换（FFT）是一种高效的算法，用于将时域信号转换为频域表示。本预处理器提供了完整的FFT变换功能，包括正向变换、逆变换、频谱分析等，支持多种输出格式和窗函数应用。

## 目录结构

```
preprocessor_FFT/
├── run.py                      # 【必需】标准运行接口
├── model.py                    # 【必需】核心算法逻辑
├── config.json                 # 【必需】默认参数配置
├── meta.json                   # 【必需】元信息
├── README.md                   # 【必需】使用说明
├── requirements.txt            # 【必需】依赖列表
├── make.py                     # 【可选】数据生成脚本
├── data/                       # 【必需】示例输入数据
│   └── example_input.npy
└── assets/                     # 【自动生成】FFT效果展示图
    └── fft_analysis_signal_*.png
```

## 依赖项

```bash
pip install numpy scipy matplotlib
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
    "signal": np.ndarray,             # 输入时域信号数据（必需）
    "fft_type": str,                  # FFT类型（可选，默认"fft"）
    "return_format": str,             # 返回格式（可选，默认"complex"）
    "window": str,                    # 窗函数类型（可选，默认"hann"）
    "sampling_rate": float,           # 采样率（可选，默认1000.0）
    "mode": str                       # 运行模式（可选，默认"transform"）
}
```

#### 参数说明

- **signal**: 输入信号数组
  - 实数信号: 一维数组 `(N,)`
  - 复数信号: 一维复数数组 `(N,)` 或 二维数组 `(2, N)` (实部,虚部)
  
- **fft_type**: FFT变换类型
  - `"fft"`: 标准FFT变换
  - `"rfft"`: 实数信号FFT（只返回正频率）
  - `"ifft"`: 逆FFT变换
  - `"irfft"`: 实数信号逆FFT变换

- **return_format**: 输出格式
  - `"complex"`: 复数形式（默认）
  - `"magnitude"`: 幅度谱
  - `"phase"`: 相位谱
  - `"power"`: 功率谱
  - `"magnitude_phase"`: 幅度和相位

- **window**: 窗函数类型
  - `"hann"`: 汉宁窗（默认）
  - `"hamming"`: 汉明窗
  - `"blackman"`: 布莱克曼窗
  - `"kaiser"`: 凯泽窗
  - `"none"`: 不应用窗函数

- **mode**: 运行模式
  - `"transform"`: 变换模式 - 执行FFT变换
  - `"inverse"`: 逆变换模式 - 执行IFFT逆变换
  - `"analyze"`: 分析模式 - 进行频谱分析

#### 输出数据格式

```python
{
    "result": np.ndarray,    # FFT变换结果
    "metrics": dict,         # 频谱分析统计信息
    "log": str,             # 日志信息
    "success": bool         # 是否成功
}
```

#### metrics 包含的频谱分析信息：
- `dominant_frequency`: 主频率 (Hz)
- `frequency_resolution`: 频率分辨率 (Hz)
- `total_power`: 总功率
- `frequency_range`: 有效频率范围 (Hz)

### 3. 运行示例

#### 基本FFT变换

```python
from run import run
import numpy as np

# 创建测试信号
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# 执行FFT变换
input_data = {
    "signal": signal,
    "fft_type": "fft",
    "return_format": "magnitude",
    "sampling_rate": 1000
}

result = run(input_data)
if result["success"]:
    magnitude_spectrum = result["result"]
    print("FFT变换成功!")
```

#### 频谱分析模式

```python
# 频谱分析
input_data = {
    "signal": signal,
    "mode": "analyze",
    "sampling_rate": 1000
}

result = run(input_data)
if result["success"]:
    print(f"主频率: {result['metrics']['dominant_frequency']} Hz")
    print(f"总功率: {result['metrics']['total_power']}")
```

### 4. 测试和可视化

```bash
python run.py
```

运行测试将：
- 加载示例数据进行FFT变换
- 测试不同的返回格式和参数
- 生成时域和频域对比的可视化图像
- 保存分析结果到 `assets/` 目录

## FFT算法效果展示

### 信号类型支持

1. **实数信号**
   - 单频正弦波
   - 多频叠加信号
   - 线性调频信号(Chirp)
   - 脉冲信号
   - 各种噪声信号

2. **复数信号**
   - QPSK调制信号
   - QAM调制信号
   - FSK调制信号
   - 复数正弦波

### 可视化效果

运行测试后将在 `assets/` 目录生成可视化图像，包括：

- **时域信号**: 原始信号波形
- **信号统计**: 均值、标准差、峰峰值等
- **幅度谱**: 频域幅度分布
- **相位谱**: 频域相位分布

### 算法特点

- ✅ **高效计算**: 基于NumPy的FFT实现，计算速度快
- ✅ **多格式输出**: 支持复数、幅度、相位、功率等多种输出格式
- ✅ **窗函数支持**: 集成多种窗函数减少频谱泄漏
- ✅ **频谱分析**: 自动计算主频率、功率等关键参数
- ✅ **复数信号**: 完整支持I/Q信号处理
- ✅ **可视化展示**: 自动生成时频域对比图

## 配置参数

`config.json` 中的默认配置：

```json
{
    "default_fft_type": "fft",
    "default_return_format": "complex", 
    "apply_window": true,
    "default_window": "hann",
    "default_sampling_rate": 1000.0
}
```

## API接口

### 主要函数

```python
def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]
```

执行FFT变换的标准接口。

### 可视化函数

```python
def visualize_fft_analysis(signal, fft_result, sampling_rate=1000, signal_index=0, save_dir=None)
```

生成FFT分析的可视化图像。

## 注意事项

1. **信号长度**: 建议使用2的幂次长度以获得最佳性能
2. **采样率**: 确保采样率满足奈奎斯特定理
3. **窗函数**: 对于非周期信号建议使用窗函数减少频谱泄漏
4. **复数信号**: 输入复数信号时可使用一维复数数组或二维实数数组格式
5. **内存使用**: 大信号处理时注意内存占用

## 版本信息

- 版本: 1.0.0
- 作者: Algorithm Team
- 更新日期: 2024-01
    "zero_padding": int,              # 零填充长度（可选）
    "sampling_rate": float,           # 采样率（可选）
    "mode": str                       # 运行模式（必需）
}
```

**详细说明：**

1. **signal**: 输入时域信号数据
   - 数据类型：`np.ndarray`
   - 形状：任意维度，如`(batch_size, signal_length)`或`(batch_size, channels, signal_length)`
   - 数据类型：可以是实数或复数

2. **fft_type**: FFT变换类型
   - 数据类型：`str`
   - 可选值：
     - `"fft"`: 标准FFT（默认）
     - `"rfft"`: 实数FFT（仅适用于实数输入）
     - `"ifft"`: 逆FFT
     - `"irfft"`: 实数逆FFT

3. **return_format**: 输出格式
   - 数据类型：`str`
   - 可选值：
     - `"complex"`: 复数格式（默认）
     - `"magnitude"`: 幅度谱
     - `"phase"`: 相位谱
     - `"magnitude_phase"`: 幅度和相位
     - `"power"`: 功率谱
     - `"power_db"`: 功率谱（dB）
     - `"real_imag"`: 实部和虚部

4. **window**: 窗函数类型
   - 数据类型：`str`
   - 可选值：`"hann"`, `"hamming"`, `"blackman"`, `"bartlett"`, `"kaiser"`, `"none"`

5. **mode**: 运行模式
   - `"transform"`: 变换模式 - 执行前向FFT变换
   - `"inverse"`: 逆变换模式 - 执行逆FFT变换
   - `"analyze"`: 分析模式 - 进行频谱分析

#### 输出数据格式

```python
{
    "result": np.ndarray,        # FFT变换结果
    "metrics": dict,             # 频谱分析信息
    "log": str,                 # 日志信息
    "success": bool             # 是否成功
}
```

**不同模式的输出内容：**

1. **transform模式**：
```python
{
    "result": fft_array,             # FFT变换结果
    "metrics": {},                   # 空字典
    "log": "FFT变换 - transform模式\n输入信号形状: (100, 1024)\nFFT类型: fft\n返回格式: magnitude",
    "success": True
}
```

2. **analyze模式**：
```python
{
    "result": fft_array,             # FFT变换结果
    "metrics": {
        "dominant_frequency": 85.93, # 主频率(Hz)
        "total_power": 512.0,        # 总功率
        "frequency_resolution": 0.977, # 频率分辨率(Hz)
        "spectral_centroid": 125.5,  # 谱重心(Hz)
        "spectral_bandwidth": 45.2,  # 谱带宽(Hz)
        "spectral_flatness": 0.234   # 谱平坦度
    },
    "log": "FFT变换 - analyze模式\n...",
    "success": True
}
```

#### 使用示例

```python
import numpy as np
from run import run

# 加载测试数据
signal_data = np.load("data/example_input.npy")  # shape: (130, 2, 1024)

# 1. 基本FFT变换（取实部作为输入）
input_data = {
    "signal": signal_data[0, 0, :],  # 单个信号
    "fft_type": "fft",
    "return_format": "magnitude",
    "mode": "transform"
}
result = run(input_data)
print(f"FFT输出形状: {result['result'].shape}")

# 2. 应用窗函数的FFT
input_data = {
    "signal": signal_data[0, 0, :],
    "window": "hann",
    "return_format": "power",
    "mode": "transform"
}
result = run(input_data)

# 3. 频谱分析模式
input_data = {
    "signal": signal_data[0, 0, :],
    "sampling_rate": 1000.0,  # 1kHz采样率
    "mode": "analyze"
}
result = run(input_data)
print(f"主频率: {result['metrics']['dominant_frequency']:.2f} Hz")
print(f"谱重心: {result['metrics']['spectral_centroid']:.2f} Hz")

# 4. 处理复数信号
complex_signal = signal_data[0, 0, :] + 1j * signal_data[0, 1, :]
input_data = {
    "signal": complex_signal,
    "return_format": "magnitude_phase",
    "mode": "transform"
}
result = run(input_data)

# 5. 逆变换
# 先进行正向变换
fft_result = np.fft.fft(signal_data[0, 0, :])
# 再进行逆变换
input_data = {
    "signal": fft_result,
    "fft_type": "ifft",
    "mode": "inverse"
}
recovered = run(input_data)
```

### 3. 配置参数

可以通过修改`config.json`文件调整算法参数：

```json
{
    "default_fft_type": "fft",           # 默认FFT类型
    "default_return_format": "complex",  # 默认返回格式
    "apply_window": true,                # 是否应用窗函数
    "normalize_output": false,           # 是否归一化输出
    "default_window": "hann",            # 默认窗函数
    "default_sampling_rate": 1000.0,     # 默认采样率
    "frequency_analysis": {
        "compute_centroid": true,        # 计算谱重心
        "compute_bandwidth": true,       # 计算谱带宽
        "compute_flatness": true,        # 计算谱平坦度
        "power_threshold": 0.1           # 功率阈值
    },
    "feature_selection": {
        "use_magnitude": true,           # 使用幅度信息
        "use_phase": true,               # 使用相位信息
        "use_power": true,               # 使用功率信息
        "use_real_imag": false           # 使用实虚部信息
    }
}
```

## 算法原理

### FFT变换

快速傅里叶变换使用分治法将DFT的计算复杂度从O(N²)降低到O(N log N)：

```
X[k] = Σ(n=0 to N-1) x[n] * e^(-j*2π*k*n/N)
```

其中：
- `x[n]`: 时域信号
- `X[k]`: 频域信号
- `N`: 信号长度

### 支持的功能

1. **多种FFT类型**：
   - **FFT**: 标准复数FFT
   - **RFFT**: 实数FFT（利用实数信号的对称性）
   - **IFFT/IRFFT**: 逆变换

2. **窗函数**：
   - **Hann窗**: 良好的频率分辨率
   - **Hamming窗**: 较好的旁瓣抑制
   - **Blackman窗**: 优秀的旁瓣抑制
   - **Kaiser窗**: 可调参数的最优窗

3. **输出格式**：
   - **复数**: 完整的幅度和相位信息
   - **幅度谱**: |X[k]|
   - **相位谱**: ∠X[k]
   - **功率谱**: |X[k]|²

4. **频谱分析**：
   - **主频率**: 功率最大的频率分量
   - **谱重心**: 频谱的"重心"频率
   - **谱带宽**: 频谱的有效带宽
   - **谱平坦度**: 频谱的平坦程度

## 性能特点

- **时间复杂度**: O(N log N)
- **空间复杂度**: O(N)
- **实时性**: 支持实时处理
- **精度**: 基于NumPy的高精度实现
- **可扩展性**: 支持多维信号处理

## 应用场景

1. **频域分析**: 分析信号的频率成分
2. **滤波预处理**: 为频域滤波做准备
3. **特征提取**: 提取频域特征用于机器学习
4. **信号处理**: 卷积、相关运算的快速实现
5. **通信系统**: 调制解调、信道分析
6. **音频处理**: 音频频谱分析、音效处理
7. **图像处理**: 2D FFT用于图像滤波和分析

## 注意事项

1. **频率混叠**: 输入信号应满足奈奎斯特采样定理
2. **频谱泄漏**: 使用适当的窗函数可以减少频谱泄漏
3. **零填充**: 可以提高频率分辨率但不增加信息
4. **内存使用**: 大信号的FFT可能需要大量内存
5. **数值精度**: 复数运算的精度限制
6. **边界效应**: 有限长信号的边界会影响频谱
