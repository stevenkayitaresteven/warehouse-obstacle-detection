@echo off
rem One-command demo launcher for Windows (cmd or double-click).
rem Usage:  run_demo.bat            start the demo
rem         run_demo.bat --full     also install the real-image detection tab
rem         run_demo.bat --check    install and run a self-check
setlocal
cd /d "%~dp0"

set "PYEXE="
rem Prefer the Python launcher "py" with a tested version.
for %%V in (3.12 3.11 3.13 3.10) do call :try_py "py -%%V"
call :try_py "py -3"
call :try_py "python"
call :try_py "python3"
if not defined PYEXE goto :nopython

echo Using: %PYEXE%
%PYEXE% run_demo.py %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :end
echo.
echo The demo stopped with an error, code %RC%. Read the messages above.
pause
goto :end

:try_py
if defined PYEXE goto :eof
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PYEXE=%~1"
goto :eof

:nopython
echo.
echo Python 3.10 or newer was not found on this computer.
echo  1. Download Python 3.12 from https://www.python.org/downloads/
echo  2. In the installer, tick "Add python.exe to PATH".
echo  3. Open a NEW cmd window and run run_demo.bat again.
echo.
pause
set "RC=1"

:end
endlocal & exit /b %RC%
