@echo off
rem run.bat - starts Changeling Imago. Sets it up first if it needs it (setup.bat).
rem   run.bat                                       opens the app
rem   run.bat -s face.png -t clip.mp4 -o out.mp4    swaps without the app
rem Uses your NVIDIA GPU when there is one; add --execution-provider cpu to force the CPU.
setlocal
cd /d "%~dp0"

call setup.bat || goto :error

set "PROVIDER="
echo.%* | find /i "--execution-provider" >nul || (
    where nvidia-smi >nul 2>&1 && set "PROVIDER=--execution-provider cuda"
)
psdenv\Scripts\python.exe run.py %PROVIDER% %*
set "EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %EXIT_CODE%

:error
pause
exit /b 1
