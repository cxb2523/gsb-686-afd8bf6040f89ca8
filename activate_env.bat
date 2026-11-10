@echo off
rem activate_env.bat - opens a command prompt inside Changeling's environment (psdenv),
rem for running things by hand: python run.py --help
cd /d "%~dp0"
if not exist psdenv\Scripts\activate.bat (
    echo Not set up yet - running setup.bat first...
    call setup.bat || (pause & exit /b 1)
)
cmd /k "psdenv\Scripts\activate.bat"
