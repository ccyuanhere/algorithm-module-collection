# 小波去噪算法

## 项目概述
小波去噪是一个基于小波变换的信号去噪算法项目，旨在有效去除信号中的高斯噪声，同时保留信号的重要特征。该项目提供了信号生成、去噪处理、性能评估以及可视化等功能。

## 功能特性
- **多级分解**：使用 `sym8` 小波进行多级分解。
- **自适应阈值**：采用自适应阈值技术（阈值乘数为 0.3）。
- **信噪比提升**：可将信噪比提升 5 - 10dB。
- **瞬态保留**：有效保留信号中的瞬态特征。

## 项目结构
```
preprocessor_WD/
├── data/                   # 存放生成的示例信号数据
├── assets/                 # 存放可视化结果
├── make.py                 # 生成带噪声的测试信号
├── config.json             # 小波去噪的配置文件
├── model.py                # 小波去噪核心处理函数
├── run.py                  # 标准运行接口，执行去噪并进行性能评估和可视化
├── meta.json               # 项目元数据
└── requirements.txt        # 项目依赖库
```

## 安装依赖
在运行项目之前，需要安装所需的依赖库。可以使用以下命令进行安装：
```bash
pip install -r requirements.txt
```

## 使用方法

### 生成示例信号
运行 `make.py` 脚本生成带噪声的测试信号和对应的干净信号：
```bash
python make.py
```
生成的信号将保存为 `data/example_input.npy` 和 `data/example_output.npy` 文件。同时，还会生成一个 `data/example_labels.npy` 文件，用于存储标签数据（目前简单生成全 1 的标签，可根据实际需求修改）。

### 运行小波去噪算法
运行 `run.py` 脚本进行小波去噪处理，并计算性能指标，同时将结果可视化：
```bash
python run.py
```
运行后，会在 `assets` 目录下生成 `denoising_comparison.png` 和 `wavelet_coefficients.png` 文件，分别展示去噪前后信号的对比和小波系数的可视化结果。

### 配置参数
可以通过修改 `config.json` 文件来调整小波去噪的参数，例如：
```json
{
    "wavelet": "sym8",
    "level": 4,
    "threshold_mode": "soft",
    "threshold_multiplier": 0.3,
    "signal_length": "rand",
    "normalization": {
        "enabled": true,
        "method": "z-score"
    }
}
```

## 性能评估
运行 `run.py` 脚本时，会计算以下性能指标：
- **原始信噪比（Original SNR）**：去噪前信号的信噪比。
- **去噪后信噪比（Denoised SNR）**：去噪后信号的信噪比。
- **信噪比提升（SNR Improvement）**：去噪前后信噪比的差值。
- **均方根误差（RMSE）**：去噪后信号与干净信号之间的均方根误差。

这些指标会在控制台输出，方便评估去噪效果。

## 注意事项
- 标签数据目前简单生成全 1 的标签，你可以根据实际需求修改 `make.py` 中的标签生成逻辑。
- 可视化结果仅展示第一个样本的信号和小波系数，可根据需要修改代码以展示更多样本。
