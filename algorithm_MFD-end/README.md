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

```bash
python run.py
```

### 4. 自定义参数运行

```bash
# 使用自定义配置文件
python run.py --config_path custom_config.json

# 运行在训练模式
python run.py --mode train

# 批量处理多通道信号
python run.py --batch_size 32
```

## 配置参数说明

`config.json` 文件包含以下可配置参数：

- `base_threshold`: 基础检测阈值，默认值为0.6
- `threshold_variation`: 阈值随机变化范围，默认值为0.4
- `snr`: 信噪比(dB)，默认值为3.0
- `sample_rate`: 采样率(Hz)，默认值为1000
- `window_size`: 滑动窗口大小，默认值为50
- `feature_count`: 特征数量，默认值为10
- `use_machine_learning`: 是否使用机器学习，默认值为false
- `ml_threshold`: 机器学习预测阈值，默认值为0.5
- `model_path`: 模型保存路径，默认值为"models/rf_model.pkl"
- `feature_selection`: 特征选择选项
  - `use_iq`: 是否使用IQ数据，默认值为true
  - `use_amplitude`: 是否使用幅度信息，默认值为true
  - `use_phase`: 是否使用相位信息，默认值为true
  - `use_spectrum`: 是否使用频谱信息，默认值为true

## 性能指标

运行算法后，将输出以下性能指标：

- 检测概率 (Pd): 正确检测到信号的概率
- 虚警概率 (Pf): 错误地检测到信号的概率
- 信噪比 (SNR): 输入信号的信噪比
- 平均检测时间: 处理每个样本的平均时间

## 调整检测性能

### 调整虚警概率

修改`make.py`中的`false_alarm_target`参数可调整类信号噪声的比例，从而控制虚警概率：

```python
# make.py 中调整此参数
false_alarm_target = 0.05  # 控制类信号噪声样本比例，建议范围 0.01-0.1
```

### 调整检测概率

1. 修改`config.json`中的`base_threshold`参数：
   - 增大阈值会降低检测概率
   - 减小阈值会提高检测概率

2. 调整`config.json`中的`snr`参数：
   - 增大SNR会提高检测概率
   - 减小SNR会降低检测概率

## 算法特点

1. **随机阈值机制**：不再使用固定阈值，引入随机性使检测结果更符合实际场景
2. **类信号噪声处理**：通过生成类似信号的噪声样本，更真实地模拟复杂电磁环境
3. **机器学习辅助**：支持使用随机森林模型优化检测决策，提高复杂环境下的检测性能
4. **多特征融合**：结合相关系数的统计特征、频谱特征和峰值特征进行综合判断

## 注意事项

1. 该算法要求已知信号模板，适用于主用户信号已知的场景
2. 检测性能受信噪比影响较大，低信噪比环境下建议结合机器学习模式
3. 调整`false_alarm_target`时需注意：过大的值会导致虚警概率过高，过小的值则无法产生有效虚警
4. 机器学习模式需要足够的训练数据才能获得最佳性能
5. 对于实时性要求高的场景，可关闭机器学习模式以提高检测速度