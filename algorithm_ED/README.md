# 能量检测频谱感知算法

## 简介
本算法实现了基于能量检测的频谱感知功能，通过计算接收信号的能量并与预设阈值比较来判断主用户是否存在。能量检测是认知无线电系统中最常用的频谱感知方法之一，具有实现简单、计算复杂度低的特点。

## 目录结构algorithm_ED/
├── run.py                      # 标准运行接口
├── model.py                    # 核心算法逻辑
├── config.json                 # 默认参数配置
├── meta.json                   # 元信息
├── README.md                   # 使用说明
├── requirements.txt            # 依赖列表
├── make.py                     # 数据生成脚本
├── assets/                     # 静态演示图
│   ├── algorithm_structure.png
│   └── example_visualization.png
├── data/                       # 示例输入输出数据
│   ├── example_input.npy
│   ├── example_output.npy
│   └── example_labels.npy
└── models/                     # 模型文件夹
    ├── pretrained_model.pt
    └── checkpoint.pkl
## 依赖项
运行本算法需要以下依赖项：
- numpy
- scipy (可选，用于某些高级功能)

可以通过以下命令安装依赖：pip install -r requirements.txt
## 使用方法

### 1. 生成示例数据
首先需要生成示例数据用于测试：python make.py这将在`data/`目录下生成示例输入、输出和标签数据。

### 2. 运行算法
可以通过以下方式调用算法：

#### 直接调用run.pyimport numpy as np
from run import run

# 加载示例数据
input_data = {
    "signal": np.load("data/example_input.npy"),
    "labels": np.load("data/example_labels.npy"),
    "mode": "evaluate"  # 可选："train", "predict", "evaluate"
}

# 运行算法
result = run(input_data)

# 输出结果
print(f"是否成功: {result['success']}")
print(f"日志: {result['log']}")
print(f"检测结果: {result['result'][:5]}")  # 显示前5个样本的检测结果
print(f"评估指标: {result['metrics']}")
#### 作为模块导入
也可以将算法作为模块导入到其他项目中：from algorithm_ED.run import run

# 准备输入数据
signal = ...  # 您的信号数据
labels = ...  # 标签数据（可选）

input_data = {
    "signal": signal,
    "labels": labels,
    "mode": "predict"
}

# 运行算法
result = run(input_data)
### 3. 配置参数
可以通过修改`config.json`文件来调整算法参数：{
    "window_size": 1024,           # 能量计算窗口大小
    "threshold_factor": 1.5,       # 阈值因子，乘以平均能量作为检测阈值
    "use_squared_energy": true,    # 是否使用信号幅度的平方计算能量
    "feature_selection": {
        "use_iq": true,            # 是否使用IQ数据
        "use_amplitude": true,     # 是否使用幅度信息
        "use_phase": true,         # 是否使用相位信息
        "use_spectrum": true       # 是否使用频谱信息
    }
}
## 算法原理
能量检测算法的基本原理是：
1. 将接收信号分成若干个时间窗口
2. 计算每个窗口内信号的能量（通常是信号幅度的平方和）
3. 将计算得到的能量与预设阈值进行比较
4. 如果能量高于阈值，则判定为主用户存在；否则判定为空闲

阈值的选择是能量检测的关键，过高的阈值会导致漏检概率增加，而过低的阈值会导致虚警概率增加。在实际应用中，阈值通常基于噪声功率估计或历史数据自适应调整。

## 性能评估
算法支持计算以下评估指标：
- 准确率(Accuracy)：正确检测的样本比例
- 精确率(Precision)：预测为正样本中实际为正样本的比例
- 召回率(Recall)：实际为正样本中被正确预测的比例
- F1分数(F1 Score)：精确率和召回率的调和平均值
- 虚警概率(False Alarm Rate)：将空闲频谱误判为主用户存在的概率
- 检测概率(Detection Rate)：正确检测到主用户存在的概率

这些指标对于评估算法在不同信噪比条件下的性能非常重要。
