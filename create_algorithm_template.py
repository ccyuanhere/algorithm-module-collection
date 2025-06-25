import os
import json
from pathlib import Path

def create_algorithm_template(algorithm_name: str):
    """
    创建算法模板目录结构
    
    Args:
        algorithm_name (str): 算法名称，如 'generator_AIS'
    """
    # 创建主目录
    base_dir = Path(algorithm_name)
    base_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    (base_dir / "assets").mkdir(exist_ok=True)
    (base_dir / "data").mkdir(exist_ok=True)
    (base_dir / "models").mkdir(exist_ok=True)
    
    # 创建基本文件
    files = [
        "run.py",
        "model.py",
        "config.json",
        "meta.json",
        "README.md",
        "requirements.txt",
        "data_generator.py"
    ]
    
    for file in files:
        (base_dir / file).touch()
    
    print(f"已创建算法模板目录: {algorithm_name}")
    print("目录结构:")
    print(f"{algorithm_name}/")
    print("├── run.py")
    print("├── model.py")
    print("├── config.json")
    print("├── meta.json")
    print("├── README.md")
    print("├── requirements.txt")
    print("├── data_generator.py")
    print("├── assets/")
    print("├── data/")
    print("└── models/")

if __name__ == "__main__":
    # 创建三个算法模板
    algorithms = [
        "generator_AIS",
        "preprocessor_normalize",
        "algorithm_knn"
    ]
    
    for algo in algorithms:
        create_algorithm_template(algo)
        print("\n" + "="*50 + "\n") 