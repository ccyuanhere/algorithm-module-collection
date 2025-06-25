import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
from colorama import init, Fore, Back, Style

# 初始化 colorama 支持 Windows 终端颜色
init()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def info_pass(text: str):
    logger.info(f"{Back.GREEN}{Fore.BLACK}{text}{Style.RESET_ALL}")

def warn_fail(text: str):
    logger.warning(f"{Back.YELLOW}{Fore.RED}{text}{Style.RESET_ALL}")

def debug_pass(text: str):
    logger.debug(f"{Back.GREEN}{Fore.BLACK}{text}{Style.RESET_ALL}")

def warn_detail(text: str):
    logger.warning(f"{Back.YELLOW}{Fore.RED}{text}{Style.RESET_ALL}")

class StructureValidator:
    def __init__(self):
        self.required_files = [
            "run.py", "model.py", "config.json", "meta.json",
            "README.md", "requirements.txt"
        ]
        self.required_dirs = ["assets", "data", "models"]
        self.required_meta_fields = [
            "name", "author", "task", "input_type", "input_size",
            "output_type", "output_size", "description", "version",
            "tags", "dependencies"
        ]
        self.checked_dirs: Set[str] = set()

    def find_algorithm_dirs(self, root_dir: str) -> List[str]:
        algorithm_dirs = []

        def is_algorithm_dir(dir_path: str) -> bool:
            return all((Path(dir_path) / f).exists() for f in ["run.py", "model.py", "config.json", "meta.json"])

        def scan_directory(current_dir: str):
            if current_dir in self.checked_dirs:
                return
            self.checked_dirs.add(current_dir)

            if is_algorithm_dir(current_dir):
                algorithm_dirs.append(current_dir)
                return

            try:
                for item in os.listdir(current_dir):
                    item_path = Path(current_dir) / item
                    if item_path.is_dir() and not item.startswith(('.', '_')):
                        scan_directory(str(item_path))
            except Exception as e:
                logger.warning(f"无法扫描目录 {current_dir}: {str(e)}")

        scan_directory(root_dir)
        return algorithm_dirs

    def check_directory_structure(self, algorithm_dir: str) -> Dict[str, Any]:
        results = {
            "directory": algorithm_dir,
            "status": "passed",
            "missing_files": [],
            "missing_dirs": [],
            "meta_issues": [],
            "data_issues": [],
            "assets_issues": []
        }

        base = Path(algorithm_dir)

        for file in self.required_files:
            if not (base / file).exists():
                results["missing_files"].append(file)
                results["status"] = "failed"

        for dir_name in self.required_dirs:
            dir_path = base / dir_name
            logger.debug(f"正在检查目录: {dir_path}")
            if not dir_path.exists() or not dir_path.is_dir():
                results["missing_dirs"].append(dir_name)
                results["status"] = "failed"
                logger.debug(f"目录不存在或不是目录: {dir_path}")

        meta_path = base / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)

                for field in self.required_meta_fields:
                    if field not in meta_data:
                        results["meta_issues"].append(f"缺少字段: {field}")
                        results["status"] = "failed"

                if not isinstance(meta_data.get("input_type", []), list):
                    results["meta_issues"].append("input_type 必须是列表")
                    results["status"] = "failed"
                if not isinstance(meta_data.get("output_type", []), list):
                    results["meta_issues"].append("output_type 必须是列表")
                    results["status"] = "failed"
                if not isinstance(meta_data.get("tags", []), list):
                    results["meta_issues"].append("tags 必须是列表")
                    results["status"] = "failed"

            except json.JSONDecodeError as e:
                results["meta_issues"].append(f"meta.json 格式错误: {e}")
                results["status"] = "failed"

        data_dir = base / "data"
        if data_dir.exists() and data_dir.is_dir():
            try:
                example_files = [f for f in data_dir.iterdir() if f.name.startswith("example_")]
                if not example_files:
                    results["data_issues"].append("缺少示例数据文件")
                    results["status"] = "failed"
            except Exception as e:
                logger.warning(f"检查 data 目录出错: {e}")

        assets_dir = base / "assets"
        if assets_dir.exists() and assets_dir.is_dir():
            try:
                image_files = [f for f in assets_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".svg")]
                if not image_files:
                    results["assets_issues"].append("缺少可视化图片资源")
                    results["status"] = "failed"
            except Exception as e:
                logger.warning(f"检查 assets 目录出错: {e}")

        return results

    def validate_all(self, root_dir: str) -> List[Dict[str, Any]]:
        algorithm_dirs = self.find_algorithm_dirs(root_dir)
        logger.info(f"共找到 {len(algorithm_dirs)} 个算法目录")

        all_results = []
        print('\n' + "*" * 50 + '\n')
        for algorithm_dir in algorithm_dirs:
            logger.info(f"\n开始检查: {algorithm_dir}")
            result = self.check_directory_structure(algorithm_dir)
            all_results.append(result)

            if result["status"] == "passed":
                info_pass(f"✓ {algorithm_dir} 通过结构检查")
            else:
                warn_fail(f"✗ {algorithm_dir} 未通过结构检查")

                # 文件缺失
                if result["missing_files"]:
                    warn_detail(f"  缺少文件: {', '.join(result['missing_files'])}")
                else:
                    debug_pass("  所有必要文件已存在！")

                # 目录缺失
                if result["missing_dirs"]:
                    warn_detail(f"  缺少目录: {', '.join(result['missing_dirs'])}")
                else:
                    debug_pass("  所有必要目录已存在！")

                # meta.json 问题
                if result["meta_issues"]:
                    warn_detail(f"  meta.json 问题: {', '.join(result['meta_issues'])}")
                else:
                    debug_pass("  meta.json 所有字段验证通过！")

                # 示例数据
                if result["data_issues"]:
                    warn_detail(f"  示例数据问题: {', '.join(result['data_issues'])}")
                else:
                    debug_pass("  示例数据文件存在！")

                # 可视化资源
                if result["assets_issues"]:
                    warn_detail(f"  可视化资源问题: {', '.join(result['assets_issues'])}")
                else:
                    debug_pass("  可视化图片资源存在")
            print('\n' + "*" * 50 + '\n')
        return all_results


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    current_dir = str(Path(__file__).parent)
    validator = StructureValidator()
    results = validator.validate_all(current_dir)

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = total - passed

    logger.info("\n结构检查统计：")
    logger.info(f"算法总数: {total}")
    info_pass(f"通过数量: {passed}")
    if failed > 0:
        warn_fail(f"未通过数量: {failed}")
        exit(1)

if __name__ == "__main__":
    main()
