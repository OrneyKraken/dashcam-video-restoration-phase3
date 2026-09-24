@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set SETUPTOOLS_USE_DISTUTILS=stdlib
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%PATH%
REM Repo root = the folder above this script, unless THESIS_P3_ROOT is already set.
if not defined THESIS_P3_ROOT for %%I in ("%~dp0..") do set "THESIS_P3_ROOT=%%~fI"
cd /d "%~dp0"
python p3_evaluate.py %*
