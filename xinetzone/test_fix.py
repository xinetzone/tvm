# 测试脚本：验证修复后的 tasks.py 代码

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

# 测试 ROOT 变量是否为 UNC 路径
from tasks import ROOT
print(f"=== 测试修复后的 ROOT 变量 ===")
print(f"ROOT: {ROOT}")
print(f"是否为 UNC 路径: {ROOT.as_posix().startswith('//') or ROOT.as_posix().startswith('\\\\')}")
print(f"ROOT 字符串表示: {str(ROOT)}")

# 测试 pip 任务是否能正确生成命令
print("\n=== 测试 pip 任务 ===")
ctx = MockContext()

# 由于 pip 任务比较复杂，我们可以直接测试它生成的命令
# 让我们创建一个简化版本的 pip 任务测试
def test_pip_command_generation():
    """测试 pip 任务是否能正确生成命令，而不涉及实际执行"""
    from tasks import CWD, ROOT
    
    print(f"CWD: {CWD}")
    print(f"ROOT: {ROOT}")
    
    # 模拟 pip 任务中的命令生成逻辑
    env = os.environ.copy()
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        if os.name == "nt":
            cmake_prefix = str(Path(prefix)/"Library")
        else:
            cmake_prefix = str(prefix)
        env["CMAKE_PREFIX_PATH"] = cmake_prefix + ";" + env.get("CMAKE_PREFIX_PATH", "")
    
    build_dir = "build"
    args = ["-DUSE_CUDA=OFF"]
    if args:
        cmake_args = ";".join(args)
        cmd = f"{sys.executable} -m pip install -ve {ROOT} --config-settings=build-dir={ROOT/build_dir} --config-settings=cmake.args=\"{cmake_args}\""
    else:
        cmd = f"{sys.executable} -m pip install -ve {ROOT} --config-settings=build-dir={ROOT/build_dir} "
    cmd += " --upgrade "
    
    print(f"生成的命令: {cmd}")
    # 检查命令中是否包含 UNC 路径
    if "\\\\" in cmd or "//" in cmd:
        print("❌ 命令中包含 UNC 路径！")
    else:
        print("✅ 命令中不包含 UNC 路径！")

test_pip_command_generation()

# 测试其他任务中的 ctx.cd 使用
print("\n=== 测试其他任务中的 ctx.cd 使用 ===")
print("1. config 任务: 不使用 ctx.cd，直接操作文件")
print("2. make 任务: 使用 ctx.cd(f'{ROOT}/build')")
print("3. Ninja 任务: 使用 ctx.cd(ROOT)")
print("4. pip 任务: 使用 ctx.cd(f'{ROOT}')")
print("5. pull 任务: 使用 os.chdir(ROOT)")
print("6. all 任务: 使用 ctx.cd(ROOT)")

# 验证这些路径是否为 UNC 路径
test_paths = [
    ROOT, 
    ROOT / "build", 
    ROOT / "3rdparty" / "tvm-ffi",
    ROOT / "3rdparty"
]

print("\n=== 验证关键路径是否为 UNC 路径 ===")
all_good = True
for path in test_paths:
    is_unc = path.as_posix().startswith('//') or path.as_posix().startswith('\\\\')
    status = "❌" if is_unc else "✅"
    print(f"{status} {path}: {'UNC 路径' if is_unc else '本地路径'}")
    if is_unc:
        all_good = False

if all_good:
    print("\n🎉 所有测试路径都是本地路径，修复成功！")
else:
    print("\n❌ 仍有路径是 UNC 路径，修复失败！")
