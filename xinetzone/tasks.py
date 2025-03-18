'''暂时仅仅提供 Linux 平台
'''
import os
import logging
import shutil
# from subprocess import Popen, PIPE
import inspect
if not hasattr(inspect, 'getargspec'): # 修复
    inspect.getargspec = inspect.getfullargspec
from pathlib import Path
import sys
from invoke import task
from taolib.flows.tasks import sites
FILE = Path(__file__).resolve() # 当前文件路径
HOME = FILE.parent # 当前目录
LOG = HOME/"logs" # 日志目录
LOG.mkdir(exist_ok=True)
ROOT = Path("..").resolve() # 获取 TVM 根目录

@task
def init(ctx,
         name='xinetzone',
         email='xinzone@outlook.com'):
    # 安装这些最小的共享库（Linux）
    ctx.run('sudo apt-get update')
    package_cmd = ('sudo apt-get install -y git '
                   'gcc g++ libtinfo-dev zlib1g-dev libzstd-dev '
                   'build-essential cmake make libedit-dev libxml2-dev')
    ctx.run(package_cmd)
    ctx.run('sudo apt install clang clangd llvm liblldb-dev')
    # Git
    ctx.run(f'git config user.name {name}')
    ctx.run(f'git config user.eamil {email}')
    ctx.run('git submodule init')
    ctx.run('git submodule update')
    # ctx.run(f"{sys.executable} -m pip install pdm")

@task
def config(ctx, cuda=False, BUILD_TYPE="RelWithDebInfo"):
    """配置 TVM 编译选项
    
    Args:
        cuda: 是否使用 CUDA
        BUILD_TYPE: 控制默认的编译选项（可选值：Release、Debug、RelWithDebInfo）
    """
    BUILD = f'{ROOT}/build'
    if Path(BUILD).exists:
        ctx.run(f'rm -rf {BUILD}')
        # shutil.rmtree(BUILD, ignore_errors=True)
    ctx.run(f'mkdir {BUILD}')
    # ctx.run(f'export VTA_HW_PATH={ROOT}/3rdparty/vta-hw')
    origin = f'{ROOT}/cmake/config.cmake'
    target = f'{ROOT}/build/config.cmake'
    with open(origin) as fp:
        content = fp.read()
    # import torch
    # torch_dynamic_library_dir = Path(torch.__file__).parent
    with open(target, 'w') as fp:
        # 控制默认的编译选项（可选值：Release、Debug、RelWithDebInfo）
        content += f'set(CMAKE_BUILD_TYPE {BUILD_TYPE})\n'

        # LLVM 是编译器端的必需依赖项
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")')
        content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM ON)')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config --ignore-libllvm")')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config-12")')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config-14")')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config --link-static")')
        
        # HIDE_PRIVATE_SYMBOLS 配置选项，启用了 -fvisibility=hidden 标志。
        # 此标志有助于防止 TVM 和 PyTorch 之间潜在的符号冲突。
        # 这些冲突是由于框架使用了不同版本的 LLVM 所导致的。
        content += 'set(HIDE_PRIVATE_SYMBOLS ON)\n'

        # Ccache 编译器包装器 可能有助于减少 TVM 的构建时间
        content = content.replace("set(USE_CCACHE OFF)", "set(USE_CCACHE AUTO)")

        # GPU SDK，按需启用
        if cuda:
            content = content.replace('set(USE_CUDA OFF)', 'set(USE_CUDA ON)')
            # content = content.replace('set(USE_VULKAN OFF)', 'set(USE_VULKAN ON)')
            # content = content.replace('set(USE_OPENCL OFF)', 'set(USE_OPENCL ON)')
            # content = content.replace('set(USE_METAL OFF)', 'set(USE_METAL ON)')

            # cuBLAS、cuDNN、cutlass 支持，按需启用
            content = content.replace('set(USE_CUDNN OFF)', 'set(USE_CUDNN ON)')
            content = content.replace('set(USE_CUBLAS OFF)', 'set(USE_CUBLAS ON)')
            # content = content.replace('set(USE_CUTLASS OFF)', 'set(USE_CUTLASS ON)')

            # 与 FlashInfer 相关的配置，需要支持计算能力为 80;86;89;90 的 CUDA
            # content = content.replace('set(USE_FLASHINFER OFF)', 'set(USE_FLASHINFER ON)')

        # 其他配置
        # content = content.replace('set(USE_MICRO OFF)', 'set(USE_MICRO ON)')
        # content = content.replace("set(USE_UMA OFF)", "set(USE_UMA ON)")
        content = content.replace("set(USE_MSC OFF)", "set(USE_MSC ON)")
        
        fp.write(content)

@task
def make(ctx):
    with ctx.cd(f'{ROOT}/build'):
        ctx.run('cmake ..')
        ctx.run("cmake --build . --parallel $(nproc)")
        # ctx.run("make cython3 -j$(nproc)")

@task
def install(ctx):
    with ctx.cd(f'{ROOT}/xinetzone'):
        ctx.run('pdm install')
        # ctx.run('pdm install -G doc')

def unlink(dst_dir):
    if dst_dir.is_symlink():
        os.unlink(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)

@task
def pull(ctx):
    '''拉取最新 TVM 文档内容'''
    os.chdir(ROOT) # 切换到上级目录
    logging.info(f"进入 {os.getcwd()}，并更新文档内容")
    dst_doc_dir = HOME/"doc/docs" # TVM 目标文档
    if dst_doc_dir.exists():
        shutil.rmtree(dst_doc_dir)
    ctx.run(f'cp -r {ROOT/"docs"} {dst_doc_dir}')
    # shutil.rmtree(f"{dst_doc_dir}/reference/api/python")
    # ctx.run(f'cp -r {HOME/"vta/tutorials"} {dst_doc_dir/"topic/vta/tutorials"}')
    # ctx.run(f'cp -r {HOME/"gallery/tutorial"} {dst_doc_dir/"tutorial"}')
    # ctx.run(f'cp -r {HOME/"gallery/how_to/*"} {dst_doc_dir/"how_to"}')

@task
def ln_env(ctx,
           root="/media/pc/data/lxw/ai/tvm",
           target="/media/pc/data/tmp/cache/conda/envs/xi",
           python_version="3.12"):
    '''将 TVM 库添加到 Python 环境
    '''
    # so_files = ["libtvm_runtime.so", "libvta_fsim.so", "libtvm.so"]
    # for so_file in so_files:
    #     ctx.run(f'ln -s {root}/{so_file} {target}/{so_file}')
    ctx.run(f'ln -s {root}/python/tvm '
            f'{target}/lib/python{python_version}/site-packages/tvm')
    ctx.run(f'ln -s {root}/vta/python/vta '
            f'{target}/lib/python{python_version}/site-packages/vta')
    ctx.run(f'ln -s {root}/3rdparty '
            f'{target}/lib/3rdparty')

# @task
# def pdm_doc(ctx, cmd=""):
#     '''文档管理'''
#     if cmd:
#         ctx.run(f"pdm run invoke doc.{cmd}")
#     else:
#         ctx.run(f"pdm run invoke doc")
        
namespace = sites(source=f"{HOME}/doc/", target=f'{HOME}/_build/html')
namespace.add_task(init)
namespace.add_task(config)
namespace.add_task(make)
namespace.add_task(install)
namespace.add_task(pull)
namespace.add_task(ln_env)
# namespace.add_task(pdm_doc) # PDM 管理文档
