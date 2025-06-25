import os
from pathlib import Path

# 定义数据文件扩展名，不进行合并
DATA_EXTENSIONS = {'.npy', '.npz', '.pt', '.pkl', '.jpg', '.png', '.svg', '.jpeg', '.bin'}

def should_skip(file: Path, excluded_files: set) -> bool:
    return (
        file.suffix.lower() in DATA_EXTENSIONS or
        file.name in excluded_files
    )

def display_directory_structure(base_dir: Path, excluded_files: set) -> str:
    structure_lines = []

    for root, dirs, files in os.walk(base_dir):
        level = Path(root).relative_to(base_dir).parts
        indent = '    ' * len(level)
        structure_lines.append(f"{indent}{Path(root).name}/")
        for file in files:
            if file not in excluded_files:
                structure_lines.append(f"{indent}    {file}")

    return '\n'.join(structure_lines)

def merge_all_files(base_dir: Path):
    script_name = Path(__file__).name
    output_filename = base_dir.name + ".txt"
    output_path = base_dir / output_filename

    excluded_files = {script_name, output_filename}

    with open(output_path, 'w', encoding='utf-8') as output_file:
        # 写入目录结构
        output_file.write("目录结构:\n")
        output_file.write("="*80 + "\n")
        output_file.write(display_directory_structure(base_dir, excluded_files))
        output_file.write("\n\n")

        # 合并文件内容
        for file in base_dir.rglob("*"):
            if file.is_file() and not should_skip(file, excluded_files):
                rel_path = file.relative_to(base_dir)
                try:
                    content = file.read_text(encoding='utf-8', errors='ignore')
                except Exception as e:
                    content = f"<<读取失败: {e}>>"

                output_file.write(f"\n{'='*80}\n# 文件: {rel_path}\n{'='*80}\n")
                output_file.write(content)
                output_file.write("\n\n")

    print(f"\n✅ 所有非数据文件已合并到: {output_path}")

if __name__ == "__main__":
    merge_all_files(Path.cwd())
