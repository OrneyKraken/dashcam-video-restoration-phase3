@echo off
REM Same environment as run_eval.bat: RVRT JIT-compiles a CUDA extension on
REM first use, which needs MSVC (vcvarsall), CUDA 12.4 and ninja on PATH.
REM Without this, rvrt is silently skipped with "Ninja is required".
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set SETUPTOOLS_USE_DISTUTILS=stdlib
set DISTUTILS_USE_SDK=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%APPDATA%\Python\Python310\Scripts;%PATH%
REM Repo root = the folder above this script, unless THESIS_P3_ROOT is already set.
if not defined THESIS_P3_ROOT for %%I in ("%~dp0..") do set "THESIS_P3_ROOT=%%~fI"
REM On the original machine C: was full, so nvcc could not write its
REM intermediate C++ files to the default %TEMP% ("No space left on device").
REM Keep the build under the repo instead of %TEMP%.
set "TORCH_EXTENSIONS_DIR=%THESIS_P3_ROOT%\.build\torch_ext"
set "TMP=%THESIS_P3_ROOT%\.build\tmp"
set "TEMP=%THESIS_P3_ROOT%\.build\tmp"
if not exist "%TMP%" mkdir "%TMP%"
cd /d "%~dp0"
python make_qualitative.py %*
