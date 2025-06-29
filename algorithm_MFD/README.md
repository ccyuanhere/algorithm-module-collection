# Matched Filter Detection

利用匹配滤波原理检测接收信号中主用户信号的存在。

## 使用说明

执行`run.py`，传入包含信号数据的字典以启动检测流程。

示例输入格式：
```python
{
    "signal": np.ndarray
}
```

输出为信号存在检测结果（0或1）。
