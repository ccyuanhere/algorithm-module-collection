# 循环平稳检测算法 (CFD)

这是一个基于循环平稳特性的频谱感知算法，用于检测信号的存在与否。该算法利用信号的循环平稳特性，通过计算循环谱密度并提取特征，使用随机森林分类器进行检测。

## 目录结构
algorithm_CFD-end/
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
    └── cfd_model.pkl       # 预训练模型
## 安装依赖

在使用该算法前，需要安装所需的依赖包：
pip install -r requirements.txt
## 使用方法

### 1. 生成示例数据

运行以下命令生成示例数据：
python make.py
这将在`data/`目录下生成三个文件：
- `example_input.npy`：示例输入信号
- `example_output.npy`：示例输出结果
- `example_labels.npy`：示例标签

### 2. 运行算法

使用以下命令运行算法：
python run.py
这将加载示例数据并运行算法，输出性能指标，包括检测概率和虚警概率。

### 3. 自定义使用

您可以在自己的代码中导入并使用该算法：
from run import run
import numpy as np

# 准备输入数据
signal = np.load('data/example_input.npy')
labels = np.load('data/example_labels.npy')

input_data = {
    "signal": signal,
    "labels": labels,
    "mode": "evaluate"  # 可选："train", "predict", "evaluate"
}

# 运行算法
result = run(input_data)

# 处理结果
if result['success']:
    print("检测结果:", result['result'])
    print("性能指标:", result['metrics'])
    print("日志:", result['log'])
else:
    print("运行失败:", result['log'])
## 算法说明

该算法基于信号的循环平稳特性，主要包含以下步骤：

1. **循环谱密度计算**：计算信号的循环谱密度，捕获信号的周期性特性
2. **特征提取**：从循环谱密度中提取统计特征
3. **模型训练**：使用随机森林分类器进行训练
4. **信号检测**：使用训练好的模型进行信号检测

## 性能指标

运行算法后，将输出以下性能指标：
- **检测概率**：正确检测到信号的概率，应接近1但不为1
- **虚警概率**：错误检测到信号的概率，应接近0但不为0
- **准确率**：分类正确的样本比例
- **精确率**：预测为正例的样本中实际为正例的比例
- **召回率**：实际为正例的样本中被预测为正例的比例
- **F1分数**：精确率和召回率的调和平均

## 配置参数

可以通过修改`config.json`文件来调整算法参数：
- `sample_rate`：采样率
- `nfft`：FFT点数
- `alpha_min`和`alpha_max`：循环频率范围
- `alpha_steps`：循环频率步数
- `detection_threshold`：检测阈值
- `feature_selection`：特征选择配置
