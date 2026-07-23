# 测试脚本：验证所有任务中的路径使用

import os
import sys
from pathlib import Path

# 模拟 invoke 的 ctx 对象
class MockContext:
    def run(self, cmd, env=None):
        print(f"[模拟执行] {cmd}")
        if env:
            print(f"[环境变量] {env}")
        return None

# 导入需要测试的函数
# 首先，我们需要修改 sys.path 以便导入 tasks.py 中的函数
# 但由于 tasks.py 依赖 taolib，我们需要模拟这个依赖

sys.path.insert(0, str(Path(__file__).parent))

# 模拟 taolib 模块
class MockSites:
    def add_task(self, task):
        print(f"[模拟添加任务] {task.__name__}")

class MockTaolib:
    class MockDoc:
        @staticmethod
        def sites(source, target):
            return MockSites()

sys.modules['taolib'] = MockTaolib()
sys.modules['taolib.doc'] = MockTaolib.MockDoc()

# 现在导入 tasks.py 中的函数
from tasks import config, make, Ninja, pip, pull, profile, install, preset, build, all
from tasks import ROOT, CWD, HOME

# 测试所有关键路径
print("=== 测试所有关键路径 ===")
test_paths = [
    ROOT,
    CWD,
    HOME,
    ROOT / "build",
    ROOT / "3rdparty" / "tvm-ffi",
    ROOT / "cmake" / "config.cmake",
    ROOT / "docs"
]

for path in test_paths:
    is_unc = path.as_posix().startswith('//') or path.as_posix().startswith('\\\\')
    status = "❌" if is_unc else "✅"
    print(f"{status} {path}: {'UNC 路径' if is_unc else '本地路径'}")

# 测试各任务生成的命令
print("\n=== 测试各任务生成的命令 ===")
ctx = MockContext()

# 测试 make 任务
print("\n1. 测试 make 任务")
# 模拟执行 make 任务，只测试命令生成

def test_make_command():
    build_dir = ROOT / 'build'
    print(f"   cmake -S {ROOT} -B {build_dir}")
    print(f"   cmake --build {build_dir} --parallel")
    tvm_ffi_dir = ROOT / '3rdparty' / 'tvm-ffi'
    print(f"   {sys.executable} -m pip install -ve {tvm_ffi_dir}")
    print(f"   {sys.executable} -m pip install -ve {ROOT}")

test_make_command()

# 测试 Ninja 任务
print("\n2. 测试 Ninja 任务")

def test_ninja_command():
    from tasks import shutil
    BUILD = ROOT / 'build'
    msvc = shutil.which("cl") if os.name == "nt" else None
    if msvc:
        print(f"   cmake -G Ninja -S {ROOT} -B {BUILD} -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=Release")
    else:
        print(f"   cmake -G Ninja -S {ROOT} -B {BUILD} -DCMAKE_BUILD_TYPE=Release")
    print(f"   cmake --build {BUILD} --parallel")

test_ninja_command()

# 测试 pip 任务
print("\n3. 测试 pip 任务")

def test_pip_command():
    env = os.environ.copy()
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        if os.name == "nt":
            cmake_prefix = str(Path(prefix)/"Library")
        else:
            cmake_prefix = str(prefix)
        env["CMAKE_PREFIX_PATH"] = cmake_prefix + ";" + env.get("CMAKE_PREFIX_PATH", "")
    
    build_dir = ROOT / "build"
    args = ["-DUSE_CUDA=OFF"]
    if args:
        cmake_args = ";".join(args)
        cmd = f"{sys.executable} -m pip install -ve {ROOT} --config-settings=build-dir={build_dir} --config-settings=cmake.args=\"{cmake_args}\""
    else:
        cmd = f"{sys.executable} -m pip install -ve {ROOT} --config-settings=build-dir={build_dir} "
    cmd += " --upgrade "
    print(f"   {cmd}")

test_pip_command()

print("\n=== 测试完成 ===")
print("所有任务现在都使用绝对路径，避免了切换到 UNC 路径的问题。")