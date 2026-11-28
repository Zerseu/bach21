@echo off
rem ---------------------------------------------------------------------------------------------
rem Makes sure the project's own virtual environment exists and has every package that
rem requirements.txt lists. The environment lives in your user folder, in
rem %USERPROFILE%\.venvs\bach21 (for example C:\Users\<you>\.venvs\bach21), not in the project
rem folder, and nothing is installed into your system Python. Called by train_bach_violin.bat and
rem generate_bach_violin.bat; on success it sets VENV_PYTHON to the environment's python.exe.
rem
rem Optional: set BACH21_TORCH_INDEX_URL to install a GPU build of PyTorch, for example
rem   set BACH21_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
rem (the default from PyPI is the CPU build).
rem Optional: set BACH21_VENV_DIR to put the environment somewhere else.
rem Delete the environment's folder to start over.
rem ---------------------------------------------------------------------------------------------

for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
if defined BACH21_VENV_DIR (
    set "VENV_DIR=%BACH21_VENV_DIR%"
) else (
    set "VENV_DIR=%USERPROFILE%\.venvs\bach21"
)
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS=%PROJECT_DIR%\requirements.txt"
set "STAMP=%VENV_DIR%\requirements.installed"

if exist "%VENV_PYTHON%" goto :check

call :find_python
if errorlevel 1 exit /b 1
echo Creating the virtual environment in %VENV_DIR% ...
%BASE_PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_failed

:check
rem Up to date when requirements.txt is unchanged since the last install and the packages import.
if not exist "%STAMP%" goto :install
fc /b "%REQUIREMENTS%" "%STAMP%" >nul 2>nul
if errorlevel 1 goto :install
"%VENV_PYTHON%" -c "import music21, torch, numpy, tqdm, matplotlib, pandas, scipy" >nul 2>nul
if errorlevel 1 goto :install
exit /b 0

:install
echo Installing the dependencies from requirements.txt into %VENV_DIR% ...
echo (the first time this downloads several hundred MB, mostly PyTorch)
if not defined BACH21_TORCH_INDEX_URL goto :install_requirements
set "TORCH_REQUIREMENT="
for /f "usebackq delims=" %%L in (`findstr /b /i /c:"torch" "%REQUIREMENTS%"`) do set "TORCH_REQUIREMENT=%%L"
if not defined TORCH_REQUIREMENT set "TORCH_REQUIREMENT=torch"
"%VENV_PYTHON%" -m pip install "%TORCH_REQUIREMENT%" --index-url "%BACH21_TORCH_INDEX_URL%"
if errorlevel 1 goto :install_failed

:install_requirements
"%VENV_PYTHON%" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 goto :install_failed
copy /y "%REQUIREMENTS%" "%STAMP%" >nul
exit /b 0

:venv_failed
echo Could not create the virtual environment in %VENV_DIR%.
exit /b 1

:install_failed
echo Installing the dependencies failed. Check the messages above and your internet connection,
echo then run the script again. Deleting the folder %VENV_DIR% starts from scratch.
exit /b 1

:find_python
rem Python 3.11 or newer is needed. Tries the "py" launcher with known-good versions first,
rem then any Python 3 it knows, then "python" on PATH.
set "BASE_PYTHON="
for %%V in (3.13 3.12 3.11 3.14 3) do (
    py -%%V -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "BASE_PYTHON=py -%%V"
        goto :found_python
    )
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "BASE_PYTHON=python"
    goto :found_python
)
echo Python 3.11 or newer was not found.
echo Install it from https://www.python.org/downloads/ and run this script again.
exit /b 1

:found_python
echo Using %BASE_PYTHON% to create the virtual environment.
exit /b 0
