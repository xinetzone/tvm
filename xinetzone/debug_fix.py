# 调试脚本：检查修复后的 ROOT 变量

import os
from pathlib import Path

# 模拟修复后的 ROOT 变量定义
CWD = Path.cwd()
FILE = Path(__file__)
HOME = FILE.parent
ROOT = CWD.parent

# 检查各种路径
print("=== 修复后的路径检查 ===")
print(f"CWD: {CWD}")
print(f"是否为 UNC 路径: {CWD.as_posix().startswith('//') or CWD.as_posix().startswith('\\\\')}")
print(f"\nFILE: {FILE}")
print(f"是否为 UNC 路径: {FILE.as_posix().startswith('//') or FILE.as_posix().startswith('\\\\')}")
print(f"\nHOME: {HOME}")
print(f"是否为 UNC 路径: {HOME.as_posix().startswith('//') or HOME.as_posix().startswith('\\\\')}")
print(f"\nROOT: {ROOT}")
print(f"是否为 UNC 路径: {ROOT.as_posix().startswith('//') or ROOT.as_posix().startswith('\\\\')}")

# 检查字符串表示
print(f"\nROOT 字符串表示: {str(ROOT)}")
print(f"ROOT.as_posix(): {ROOT.as_posix()}")

# 检查 os.getcwd()
cwd = os.getcwd()
print(f"\nos.getcwd(): {cwd}")
print(f"是否为 UNC 路径: {cwd.startswith('//') or cwd.startswith('\\\\')}")
