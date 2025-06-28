# 循环平稳特征检测频谱感知算法

## 一、概述  
本算法是基于循环平稳特征的频谱感知算法，利用信号的周期性特征进行主用户检测。通过计算信号的谱相关密度函数（SCD），并在目标循环频率处进行特征检测，实现对信号存在与否的判断。  


## 二、文件结构  
```plaintext
algorithm_EDC/
├── meta.json          # 算法元信息
├── requirements.txt   # 依赖库列表
├── make.py            # 生成测试数据脚本
├── run.py             # 算法运行脚本
├── model.py           # 算法核心逻辑脚本
├── config.json        # 算法配置文件
└── data/              # 数据存储目录
```  

## 三、环境准备  
### 1. 安装依赖库  
在项目根目录下执行以下命令安装所需的依赖库：  
```pip install -r requirements.txt
```  

## 四、生成测试数据  
### 1. 脚本说明  
make.py 脚本用于生成测试数据集，包括信号数据和对应的标签数据。  

### 2. 运行命令  
在项目根目录下执行以下命令生成测试数据：  
```python make.py
```  

### 3. 数据生成参数  
- num_segments：信号段数量，默认值为 100。  
- segment_length：每段信号长度，默认值为 2048。  
- sampling_rate：采样率，默认值为 8000 Hz。  

### 4. 生成结果  
生成的数据将保存到 data/ 目录下，包括：  
- example_input.npy：信号数据  
- example_labels.npy：标签数据  
- example_output.npy：示例输出数据  


## 五、运行算法  
### 1. 脚本说明  
run.py 脚本是算法的统一运行接口，调用 model.py 中的核心处理函数进行信号检测，并输出检测结果和性能评估指标。  

### 2. 运行命令  
在项目根目录下执行以下命令运行算法：  
```python run.py
```  

### 3. 输入数据格式  
输入数据为一个字典，包含以下字段：  
- signal：输入信号，类型为 np.ndarray，形状为 [num_segments, segment_length]。  
- labels：标签数据，类型为 np.ndarray，形状为 [num_segments]，可选参数。  
- mode：运行模式，默认值为 "evaluate"。  

### 4. 配置文件  
算法的配置参数存储在 config.json 文件中，包括：  
- fft_points：FFT 点数，默认值为 1024。  
- time_window：时域平滑窗口长度，默认值为 64。  
- freq_window：频域平滑窗口长度，默认值为 16。  
- alpha_resolution：循环频率分辨率，默认值为 0.01。  
- target_alpha：目标循环频率，默认值为 0.2。  
- detection_threshold：检测阈值，默认值为 1.5。  
- snr_range：信噪比范围，默认值为 [-20, 0]。  
- signal_type：信号类型，默认值为 "BPSK"。  
- symbol_rate：符号速率，默认值为 1000 Hz。  
- min_snr_ratio：最小信噪比比值，默认值为 0.5。  
- max_snr_ratio：最大信噪比比值，默认值为 10.0。  

### 5. 输出结果  
输出结果为一个字典，包含以下字段：  
- result：检测结果，类型为 np.ndarray，形状为 [num_segments]。  
- metrics：性能评估指标，类型为字典，包括检测概率、虚警概率、漏检概率、准确率等。  
- log：日志信息。  
- success：是否成功执行，类型为布尔值。  


## 六、算法核心逻辑  
### 1. 谱相关密度函数（SCD）估计  
model.py 中的 estimate_scd 函数使用频域平滑法（FAM）估计信号的谱相关密度函数（SCD）。  

### 2. 循环平稳特征检测器  
model.py 中的 cyclo_detector 函数是循环平稳特征检测器的核心函数，通过计算目标循环频率处的平均 SCD 幅度和背景噪声水平，得到检验统计量和信噪比比值。  

### 3. 性能评估  
model.py 中的 evaluate_performance 函数用于评估检测性能，计算检测概率、虚警概率、漏检概率、准确率等性能指标。  

### 4. 自适应阈值设置  
在有标签数据的情况下，使用 Fisher 判别法确定最佳阈值；在没有标签数据的情况下，使用固定阈值。  


## 七、注意事项  
- 请确保在运行算法前已经安装了所需的依赖库。  
- 可以根据需要修改 config.json 文件中的配置参数，以调整算法的性能。  
- 在生成测试数据时，可以修改 make.py 脚本中的参数，以生成不同规模和特性的测试数据集。