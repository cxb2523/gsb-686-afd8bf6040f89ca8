@echo off
rem setup.bat - gets Changeling ready: the psdenv virtual environment (Python 3.9 - 3.12),
rem the packages in requirements.txt and the models. Safe to run again: it only redoes what's
rem missing or changed. run.bat calls it for you, so you only need this on its own to set up
rem without opening the app.
setlocal
cd /d "%~dp0"

set "VENV_DIR=psdenv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

rem 1) The virtual environment (TensorFlow and PyTorch need Python 3.9 - 3.12).
if not exist "%VENV_PY%" (
    echo Creating the virtual environment...
    set "BASE_PY="
    for %%V in (3.11 3.12 3.10 3.9) do (
        if not defined BASE_PY (
            py -%%V -c "import sys" >nul 2>&1 && set "BASE_PY=py -%%V"
        )
    )
    if not defined BASE_PY (
        python -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "BASE_PY=python"
    )
    if not defined BASE_PY (
        echo Python 3.9 - 3.12 is needed ^(3.11 is best^). Get it from https://www.python.org/downloads/
        exit /b 1
    )
    call %%BASE_PY%% -m venv "%VENV_DIR%" || goto :error
)

rem 2) The packages, whenever requirements.txt changed since the last install.
rem    The first time downloads several GB (PyTorch, TensorFlow) - give it a while.
fc /b requirements.txt "%VENV_DIR%\.requirements" >nul 2>&1
if errorlevel 1 (
    echo Installing the packages - the first time takes a while...
    "%VENV_PY%" -m pip install --upgrade pip || goto :error
    "%VENV_PY%" -m pip install -r requirements.txt || goto :error
    copy /y requirements.txt "%VENV_DIR%\.requirements" >nul
)

rem 3) Any missing models.
"%VENV_PY%" scripts/download_models.py || goto :error

rem 4) Which GPU, and a warning if its driver is too old.
"%VENV_PY%" scripts/check_gpu.py

echo Changeling Imago is ready. Start it with run.bat
exit /b 0

:error
echo.
echo Setup failed - see the messages above. Run setup.bat again to retry; it carries on from where it stopped.
exit /b 1
