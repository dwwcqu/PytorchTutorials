import torch
import os
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
)

library_name = "dengww"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available()
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources
    print(f'sources:{sources}')
    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

PROJECT_MAJOR=0
PROJECT_MINOR=0
PROJECT_TAG=1

PROJECT_VERSION=f'{PROJECT_MAJOR}.{PROJECT_MINOR}.{PROJECT_TAG}+torch{TORCH_MAJOR}.{TORCH_MINOR}'
setup(
    name=library_name,
    version=PROJECT_VERSION,
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Example of PyTorch C++ and CUDA extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://192.168.140.75:9080/dengww/PytorchTutorials/ExtendPytorch/cutom_ops",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)