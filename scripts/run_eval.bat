@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set SETUPTOOLS_USE_DISTUTILS=stdlib
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%PATH%
set THESIS_P3_ROOT=F:\user4\thesis_p3
cd /d F:\user4\thesis_p3\scripts
python p3_evaluate.py %*
