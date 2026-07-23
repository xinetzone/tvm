"""Invoke 任务集：构建与配置 TVM。

跨平台支持（Linux/Windows），提供初始化依赖、配置构建选项、
编译安装以及文档同步等任务。保持与现有 CLI 使用方式兼容。
"""

import os
import re
import platform
import logging
import shutil
import inspect
from pathlib import Path
from typing import Optional, Union
import sys
import click
from invoke import task
from invoke.collection import Collection
from taolib.doc import sites
# 获取当前工作目录，使用映射的驱动器路径，避免 UNC 路径问题
CWD = Path.cwd()
FILE = Path(__file__)
HOME = FILE.parent # 当前目录
LOG = HOME/"logs" # 日志目录
LOG.mkdir(exist_ok=True)
# 获取 TVM 根目录，使用相对路径转换，确保使用映射的驱动器路径
ROOT = CWD.parent

# @task
# def init(ctx,
#          name: str = 'xinetzone',
#          email: str = 'xinzone@outlook.com'):
#     """初始化依赖与 Git 子模块。

#     仅在 Linux 下安装系统依赖；所有平台均初始化 Git 配置与子模块。
#     """
#     # 安装这些最小的共享库（Linux）
#     ctx.run('sudo apt-get update')
#     package_cmd = ('sudo apt-get install -y git '
#                    'gcc g++ libtinfo-dev zlib1g-dev libzstd-dev '
#                    'build-essential cmake make libedit-dev libxml2-dev')
#     ctx.run(package_cmd)
#     ctx.run('sudo apt install clang clangd llvm liblldb-dev')
#     # Git
#     ctx.run(f'git config user.name {name}')
#     ctx.run(f'git config user.eamil {email}')
#     ctx.run('git submodule init')
#     ctx.run('git submodule update')
#     # ctx.run(f"{sys.executable} -m pip install pdm")

def _cmake_get(content: str, var: str) -> Optional[str]:
    m = re.search(rf"set\({re.escape(var)}\s+([^)\n]+)\)", content)
    return m.group(1).strip() if m else None

def _cmake_set(content: str, var: str, value: Union[str, bool]) -> str:
    val = str(value)
    if not (val.startswith('"') and val.endswith('"')) and ("/" in val or \
            (os.name == "nt" and (":" in val or "\\" in val))):
        val = f'"{val}"'
    pattern = rf"set\({re.escape(var)}\s+[^)\n]+\)"
    repl = f"set({var} {val})"
    if re.search(pattern, content):
        return re.sub(pattern, repl, content)
    return content + "\n" + repl + "\n"

def _read_text(path: Union[str, Path]) -> str:
    p = Path(path)
    return p.read_text(encoding='utf-8')

def _write_text(path: Union[str, Path], content: str) -> None:
    p = Path(path)
    p.write_text(content, encoding='utf-8')

def _build_config_content(base: str,
                          build_type: str,
                          cuda: bool,
                          vta: bool,
                          vulkan: bool,
                          acl_codegen: Optional[bool],
                          acl_graph_executor: Optional[Union[bool, str]]) -> str:
    """组合并返回最终 CMake 配置文本。"""
    content = base + f'\nset(CMAKE_BUILD_TYPE {build_type})\n'
    content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM ON)')
    content += 'set(HIDE_PRIVATE_SYMBOLS ON)\n'
    content = content.replace("set(USE_CCACHE OFF)", "set(USE_CCACHE AUTO)")
    if cuda:
        content = content.replace('set(USE_CUDA OFF)', 'set(USE_CUDA ON)')
    if vulkan:
        content = content.replace('set(USE_VULKAN OFF)', 'set(USE_VULKAN ON)')
        content = content.replace('set(USE_CUDNN OFF)', 'set(USE_CUDNN ON)')
        content = content.replace('set(USE_CUBLAS OFF)', 'set(USE_CUBLAS ON)')
    content = content.replace("set(USE_MSC OFF)", "set(USE_MSC ON)")

    cur_acl = _cmake_get(content, 'USE_ARM_COMPUTE_LIB')
    cur_acl_ge = _cmake_get(content, 'USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR')
    desired_acl = (cur_acl or 'OFF') if (acl_codegen is None) else ('ON' if acl_codegen else 'OFF')
    if acl_graph_executor is None:
        desired_acl_ge: Union[str, bool] = cur_acl_ge or 'OFF'
    else:
        desired_acl_ge = 'ON' if isinstance(acl_graph_executor, bool) and acl_graph_executor else (
            'OFF' if isinstance(acl_graph_executor, bool) else str(acl_graph_executor)
        )
        if isinstance(acl_graph_executor, str):
            p = Path(acl_graph_executor.strip('"'))
            if not p.exists():
                raise ValueError(f'Invalid ACL path: {acl_graph_executor}')

    if desired_acl_ge == 'ON' and platform.machine().lower() not in ('aarch64', 'arm64', 'armv7l', 'arm'):
        raise RuntimeError('ACL runtime cannot be built on non-ARM hosts; provide path or disable')

    content = _cmake_set(content, 'USE_ARM_COMPUTE_LIB', desired_acl)
    content = _cmake_set(content, 'USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR', desired_acl_ge)
    if vta:
        content += '\nset(USE_VTA_FSIM ON)'
    return content

@task
def config(ctx,
           cuda: bool = False,
           vta: bool = False,
           vulkan: bool = False,
           BUILD_TYPE: str = "RelWithDebInfo",
           acl_codegen: Optional[bool] = None,
           acl_graph_executor: Optional[Union[bool, str]] = None):
    """配置 TVM 编译选项并生成 `build/config.cmake`。

    保持上游 `cmake/config.cmake` 的默认值，允许通过参数覆盖 CUDA/Vulkan/ACL 等选项。
    """
    build_dir = ROOT / 'build'
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    origin = ROOT / 'cmake' / 'config.cmake'
    target = build_dir / 'config.cmake'
    base = _read_text(origin)
    final = _build_config_content(base, BUILD_TYPE, cuda, vta, vulkan, acl_codegen, acl_graph_executor)
    _write_text(target, final)

@task
def make(ctx):
    """使用 CMake 构建并安装 Python 包与 FFI 扩展。"""
    build_dir = ROOT / 'build'
    # 直接在命令中使用绝对路径，避免切换到 UNC 路径
    ctx.run(f'cmake -S {ROOT} -B {build_dir}')
    ctx.run(f"cmake --build {build_dir} --parallel")
    # ctx.run("make cython3 -j$(nproc)")
    # 安装 TVM-FFI 包
    tvm_ffi_dir = ROOT / '3rdparty' / 'tvm-ffi'
    ctx.run(f"{sys.executable} -m pip install -ve {tvm_ffi_dir}")
    # 安装 TVM 包
    ctx.run(f"{sys.executable} -m pip install -ve {ROOT}")

@task
def Ninja(ctx):
    """使用 Ninja 加速构建，优先选择 MSVC（Windows）。"""
    BUILD = ROOT / 'build'
    # 使用 Ninja 加速编译，交由 CMake 选择平台合适的编译器
    # 在 Windows 优先尝试 MSVC 的 cl 编译器，避免 Conda clang+lld 链接错误
    msvc = shutil.which("cl") if os.name == "nt" else None
    if msvc:
        # MSVC 环境下显式指定 cl 编译器，并设置为 Release 单配置
        ctx.run(
            f"cmake -G Ninja -S {ROOT} -B {BUILD} "
            f"-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl "
            f"-DCMAKE_BUILD_TYPE=Release"
        )
    else:
        # 其他平台保持 Ninja，设置构建类型
        ctx.run(
            f"cmake -G Ninja -S {ROOT} -B {BUILD} -DCMAKE_BUILD_TYPE=Release"
        )
    # 统一使用 CMake 的并行参数，避免向 Ninja 传递 MSBuild 专用的 /m
    ctx.run(f"cmake --build {BUILD} --parallel")

@task
def pip(ctx,
        ensure_deps: bool = False,
        gpu: bool = False,
        generator: Optional[str] = None,
        archs: str = "80;86;89;90",
        cuda_path: Optional[str] = None):
    """通过 scikit-build-core 以 `pip install -ve .` 方式构建安装。

    可选启用 GPU 构建，并在 Windows 上显式传入生成器与 NVCC 路径。
    """
    def _ensure_deps(ctx):
        prefix = os.environ.get("CONDA_PREFIX")
        if not prefix:
            if sys.platform.startswith("linux"):
                ctx.run("sudo apt-get update")
                ctx.run("sudo apt-get install -y zlib1g-dev libzstd-dev")
                return
            raise RuntimeError("Missing CONDA_PREFIX; activate your Conda env")
        if os.name == "nt":
            inc = Path(prefix)/"Library"/"include"/"zlib.h"
            libs = [
                Path(prefix)/"Library"/"lib"/"zlib.lib",
                Path(prefix)/"Library"/"lib"/"z.lib",
            ]
        else:
            inc = Path(prefix)/"include"/"zlib.h"
            libs = list((Path(prefix)/"lib").glob("libz.*"))
        if (not inc.exists()) or (not any(p.exists() for p in libs)):
            ctx.run("conda install -y -c conda-forge zlib zstd")
    def _find_nvcc(cuda_path):
        cands = [cuda_path, os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"), "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"] if os.name == "nt" else [cuda_path, os.environ.get("CUDA_HOME"), "/usr/local/cuda"]
        for c in cands:
            if c:
                p = Path(c)/"bin"/("nvcc.exe" if os.name == "nt" else "nvcc")
                if p.exists():
                    return str(p)
        return None
    skip = os.environ.get("TVM_SKIP_DEPS", "").lower() in ("1", "true", "yes")
    if ensure_deps and not skip:
        _ensure_deps(ctx)
    env = os.environ.copy()
    prefix = os.environ.get("CONDA_PREFIX")
    cmake_prefix = (
        str(Path(prefix)/"Library") if (prefix and os.name == "nt") else (str(prefix) if prefix else "/usr")
    )
    env["CMAKE_PREFIX_PATH"] = cmake_prefix + ";" + env.get("CMAKE_PREFIX_PATH", "")
    # 使用绝对路径作为构建目录，确保是映射的驱动器路径，避免 UNC 路径
    build_dir = ROOT / "build"
    if gpu and os.name == "nt":
        env["CMAKE_ROOT"] = str(Path(sys.prefix)/"Lib"/"site-packages"/"cmake"/"data"/"share"/"cmake-4.2")
    args = []
    if gpu:
        gen = generator or ("Visual Studio 17 2022" if os.name == "nt" else None)
        if gen:
            args += ["-G", gen]
        nvcc = _find_nvcc(cuda_path)
        if nvcc:
            args += [f"-DCMAKE_CUDA_COMPILER={nvcc}"]
        args += ["-DUSE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={archs}"]
    else:
        args += ["-DUSE_CUDA=OFF"]
    if args:
        cmake_args = ";".join(args)
        # 直接在当前目录执行 pip 命令，使用绝对路径指向 TVM 根目录，避免切换到 UNC 路径
        cmd = f"{sys.executable} -m pip install -ve {ROOT} --config-settings=build-dir={build_dir} --config-settings=cmake.args=\"{cmake_args}\""
    else:
        cmd = f"{sys.executable} -m pip install -ve {ROOT} --config-settings=build-dir={build_dir} "
    cmd += " --upgrade "
    ctx.run(cmd, env=env)
    with ctx.cd(f'{ROOT}/3rdparty/tvm-ffi'):
        ctx.run(f"{sys.executable} -m pip install -ve .")

def unlink(dst_dir):
    if dst_dir.is_symlink():
        os.unlink(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)

@task
def pull(ctx):
    """拉取并同步 TVM 文档到本地扩展目录。"""
    # 不再切换工作目录，直接使用绝对路径操作文件
    logging.info(f"更新文档内容，ROOT: {ROOT}")
    dst_doc_dir = HOME/"doc/docs" # TVM 目标文档
    if dst_doc_dir.exists():
        shutil.rmtree(dst_doc_dir)
    dst_doc_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT/"docs", dst_doc_dir)
    # shutil.rmtree(f"{dst_doc_dir}/reference/api/python")
    # ctx.run(f'cp -r {HOME/"vta/tutorials"} {dst_doc_dir/"topic/vta/tutorials"}')
    # ctx.run(f'cp -r {HOME/"gallery/tutorial"} {dst_doc_dir/"tutorial"}')
    # ctx.run(f'cp -r {HOME/"gallery/how_to/*"} {dst_doc_dir/"how_to"}')

@task
def profile(ctx):
    """Conan：检测并生成 profile。"""
    ctx.run("conan profile detect --force")

@task
def install(ctx):
    """Conan：安装依赖并生成工具链。"""
    conf = "tools.cmake.cmaketoolchain:generator=Ninja"
    ctx.run(f"conan install . -c {conf} --build=missing")

@task
def preset(ctx):
    """CMake：应用 `conan-release` 预设。"""
    ctx.run("cmake --preset conan-release")

@task
def build(ctx):
    """CMake：基于预设进行构建。"""
    ctx.run("cmake --build --preset conan-release")

@task
def all(ctx):
    """执行 Conan 与 CMake 全流程：profile→install→preset→build。"""
    # 直接在命令中使用绝对路径，避免切换到 UNC 路径
    # 注意：profile, install, preset, build 任务可能需要在 ROOT 目录下执行
    # 我们需要修改这些任务，让它们接受一个 root_dir 参数，或者直接在任务内部使用 ROOT 变量
    # 目前，这些任务比较简单，我们可以直接在命令中使用绝对路径
    # 由于 profile, install, preset, build 任务都是简单的命令执行，没有复杂的逻辑，我们可以直接修改它们
    with ctx.cd(ROOT):
        profile(ctx)
        install(ctx)
        preset(ctx)
        build(ctx)

namespace = sites(source=f"{HOME}/doc", target=f'{HOME}/_build/html')
# namespace.add_task(init)
namespace.add_task(config)
namespace.add_task(make)
namespace.add_task(Ninja)
namespace.add_task(pull)
namespace.add_task(all)
namespace.add_task(pip)

# # 创建一个 doc 子命名空间，用于存放文档相关的任务
# # 这样我们就可以使用 python -m invoke doc.intl -l zh_CN 命令来调用 intl 任务
# docx = Collection('doc')

# @task
# def intl(ctx, l: str = 'zh_CN', d: Optional[str] = None):
#     """生成与更新国际化文档资源。"""
#     pot = HOME/'_build'/'gettext'
#     if not pot.exists():
#         with ctx.cd(HOME):
#             ctx.run(f"{sys.executable} -m sphinx.cmd.build -b gettext doc {pot}")
#     locale_dir = d or 'locales'
#     abs_locale = HOME/locale_dir
#     (abs_locale/ l / 'LC_MESSAGES').mkdir(parents=True, exist_ok=True)
    
#     # 直接在代码中实现 sphinx-intl update 的功能，避免使用 sphinx-intl 命令
#     import sphinx_intl.basic as sphinx_intl_basic
#     sphinx_intl_basic.update(
#         str(abs_locale),
#         str(pot),
#         (l,),  # 语言列表，需要是 tuple 类型
#         line_width=76,
#         ignore_obsolete=False,
#         jobs=0
#     )

# # 将 intl 任务添加到 doc 子命名空间中
# docx.add_task(intl)
# # 将 doc 子命名空间添加到主命名空间中
# namespace.add_collection(docx)