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
from d2py.tools.write import site
from d2py.utils.log_config import config_logging
FILE = Path(__file__).resolve() # 当前文件路径
HOME = FILE.parent # 当前目录
LOG = HOME/"logs" # 日志目录
LOG.mkdir(exist_ok=True)
ROOT = Path("..").resolve() # 获取 TVM 根目录
# 配置日志
config_logging(
    f"{LOG}/{FILE.name.removesuffix(FILE.suffix)}.log", 
    'task',
    filter_mod_names={"invoke", "h5py"}, # 过滤掉不需要记录日志的模块
)

@task
def init(ctx,
         name='xinetzone',
         email='xinzone@outlook.com'):
    # 安装这些最小的共享库（Linux）
    ctx.run('sudo apt-get update')
    package_cmd = ('sudo apt-get install -y git '
                   'gcc g++ libtinfo-dev zlib1g-dev '
                   'build-essential cmake make libedit-dev libxml2-dev')
    ctx.run(package_cmd)
    ctx.run('sudo apt install clang clangd llvm liblldb-dev')
    # Git
    ctx.run(f'git config user.name {name}')
    ctx.run(f'git config user.eamil {email}')
    ctx.run('git submodule init')
    ctx.run('git submodule update')
    ctx.run(f"{sys.executable} -m pip install pdm")

@task
def config(ctx, cuda=False):
    
    BUILD = f'{ROOT}/build'
    if Path(BUILD).exists:
        ctx.run(f'rm -rf {BUILD}')
            #shutil.rmtree(BUILD, ignore_errors=True)
    ctx.run(f'mkdir {BUILD}')
    ctx.run(f'export VTA_HW_PATH={ROOT}/3rdparty/vta-hw')
    origin = f'{ROOT}/cmake/config.cmake'
    target = f'{ROOT}/build/config.cmake'
    with open(origin) as fp:
        content = fp.read()
    # import torch
    # torch_dynamic_library_dir = Path(torch.__file__).parent
    with open(target, 'w') as fp:
        content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM ON)')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config")')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config-12")')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config-16")')
        # content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM "llvm-config --link-static")')
        content = content.replace('set(USE_VTA_FSIM OFF)', 'set(USE_VTA_FSIM ON)')
        content = content.replace('set(USE_RELAY_DEBUG OFF)', 'set(USE_RELAY_DEBUG ON)')
        content = content.replace("set(USE_PIPELINE_EXECUTOR OFF)", "set(USE_PIPELINE_EXECUTOR ON)")
        content = content.replace('set(USE_MICRO OFF)', 'set(USE_MICRO ON)')
        content = content.replace("set(USE_UMA OFF)", "set(USE_UMA ON)")
        content = content.replace("set(USE_MSC OFF)", "set(USE_MSC ON)")
        content = content.replace("set(USE_PROFILER OFF)", "set(USE_PROFILER ON)")
        # content = content.replace("set(USE_LIBTORCH OFF)", f"set(USE_LIBTORCH {torch_dynamic_library_dir})")
        if cuda:
            content = content.replace('set(USE_CUDA OFF)', 'set(USE_CUDA ON)')
            # content = content.replace('set(USE_CUDA OFF)', 'set(USE_CUDA /usr/local/cuda-12/lib64)')
            # content = content.replace('set(USE_CUDA OFF)', 'set(USE_CUDA /usr/local/cuda-12.4/lib64)')
            # content = content.replace('set(USE_CUDA OFF)', 'set(USE_CUDA /usr/local/cuda-12.4/targets/x86_64-linux/lib)')
            content = content.replace('set(USE_CUDNN OFF)', 'set(USE_CUDNN ON)')
            content = content.replace('set(USE_CUBLAS OFF)', 'set(USE_CUBLAS ON)')
            content = content.replace("set(USE_VERILATOR OFF)", "set(USE_VERILATOR ON)")
            # content = content.replace("set(USE_METAL OFF)", "set(USE_METAL ON)")
            # content = content.replace('set(USE_CUTLASS OFF)', 'set(USE_CUTLASS ON)')
            # content = content.replace("set(USE_PAPI OFF)", "set(USE_PAPI ON)")
            # content = content.replace('set(USE_NNPACK OFF)', 'set(USE_NNPACK ON)')
            # content = content.replace("set(USE_VTA_TSIM OFF)", "set(USE_VTA_TSIM ON)") 
        # PyTorch and TVM loading problem due to conflicting LLVM symbols
        # https://github.com/apache/tvm/issues/9362
        content += 'set(HIDE_PRIVATE_SYMBOLS ON)'
        fp.write(content)

@task
def make(ctx):
    with ctx.cd(f'{ROOT}/build'):
        ctx.run('cmake ..')
        ctx.run("make -j$(nproc)")
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
    ctx.run(f'cp -r {HOME/"vta/tutorials"} {dst_doc_dir/"topic/vta/tutorials"}')
    ctx.run(f'cp -r {HOME/"gallery/tutorial"} {dst_doc_dir/"tutorial"}')
    ctx.run(f'cp -r {HOME/"gallery/how_to/*"} {dst_doc_dir/"how_to"}')

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
        
namespace = site(source=f"{HOME}/doc/", target=f'{HOME}/_build/html')
namespace.add_task(init)
namespace.add_task(config)
namespace.add_task(make)
namespace.add_task(install)
namespace.add_task(pull)
namespace.add_task(ln_env)
# namespace.add_task(pdm_doc) # PDM 管理文档
