@echo off
REM Same environment as run_eval.bat: RVRT JIT-compiles a CUDA extension on
REM first use, which needs MSVC (vcvarsall), CUDA 12.4 and ninja on PATH.
REM Without this, rvrt is silently skipped with "Ninja is required".
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set SETUPTOOLS_USE_DISTUTILS=stdlib
set DISTUTILS_USE_SDK=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%APPDATA%\Python\Python310\Scripts;%PATH%
REM The C: drive is full, so nvcc cannot write its intermediate C++ files to
REM the default %TEMP% and fails with "No space left on device". Keep the
REM whole build off C:.
set TORCH_EXTENSIONS_DIR=F:\user4\thesis_p3\.build\torch_ext
set TMP=F:\user4\thesis_p3\.build\tmp
set TEMP=F:\user4\thesis_p3\.build\tmp
set THESIS_P3_ROOT=F:\user4\thesis_p3
cd /d F:\user4\thesis_p3\scripts
python make_qualitative.py %*
