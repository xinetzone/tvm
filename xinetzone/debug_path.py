# 调试脚本：检查 ROOT 变量和 ctx.cd 使用

import os
import platform
from pathlib import Path

# 模拟 tasks.py 中的 FILE 和 ROOT 定义
FILE = Path(__file__).resolve() # 当前文件路径
HOME = FILE.parent # 当前目录

# 原始定义
print("=== 原始定义 ===")
original_root = Path("..").resolve()
print(f"Path('..').resolve(): {original_root}")
print(f"是否为 UNC 路径: {original_root.as_posix().startswith('//') or original_root.as_posix().startswith('\\\\')}")

# 修复后的定义
print("\n=== 修复后的定义 ===")
if os.name == "nt":
    # 在 Windows 下，使用当前文件的父目录的父目录，确保是本地路径
    root = FILE.parent.parent
else:
    root = Path("..").resolve()
print(f"FILE.parent.parent: {root}")
print(f"是否为 UNC 路径: {root.as_posix().startswith('//') or root.as_posix().startswith('\\\\')}")

# 检查 os.getcwd()
print("\n=== 当前工作目录 ===")
cwd = os.getcwd()
print(f"os.getcwd(): {cwd}")
print(f"是否为 UNC 路径: {cwd.startswith('//') or cwd.startswith('\\\\')}")

# 检查 Path.cwd()
print("\n=== Path.cwd() ===")
path_cwd = Path.cwd()
print(f"Path.cwd(): {path_cwd}")
print(f"是否为 UNC 路径: {path_cwd.as_posix().startswith('//') or path_cwd.as_posix().startswith('\\\\')}")

# 检查各种路径转换
print("\n=== 路径转换测试 ===")
# 测试相对路径转换
rel_path = Path("..")
print(f"Path('..'): {rel_path}")
print(f"Path('..').absolute(): {rel_path.absolute()}")
print(f"Path('..').resolve(): {rel_path.resolve()}")

# 测试 UNC 路径检测
unc_test = "\\\\server\\share\\path"
print(f"\nUNC 路径测试: {unc_test}")
print(f"是否为 UNC 路径: {unc_test.startswith('//') or unc_test.startswith('\\\\')}")

# 测试 FILE.parent.parent
print(f"\nFILE.parent.parent: {FILE.parent.parent}")
print(f"类型: {type(FILE.parent.parent)}")
print(f"字符串表示: {str(FILE.parent.parent)}")
print(f"as_posix(): {FILE.parent.parent.as_posix()}")
print(f"是否为 UNC 路径: {FILE.parent.parent.as_posix().startswith('//') or FILE.parent.parent.as_posix().startswith('\\\\')}")
