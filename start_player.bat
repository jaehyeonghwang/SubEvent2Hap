@echo off
rem Laptop side: Quest playback agent for the SubEvent2Hap server on the Tailscale network.
rem Usage: start_player.bat [server-url]   (default http://desktop-nst9a4u:7860)
setlocal
cd /d "%~dp0"
set SERVER=%~1
if "%SERVER%"=="" set SERVER=http://desktop-nst9a4u:7860
if not exist .venv-player\Scripts\python.exe (
  echo [setup] creating .venv-player ...
  py -3 -m venv .venv-player || python -m venv .venv-player || goto :nopython
  .venv-player\Scripts\python -m pip install -q --upgrade pip
  .venv-player\Scripts\python -m pip install -q -r requirements-player.txt || goto :eof
)
echo Server: %SERVER%   (editor: %SERVER%/editor)
.venv-player\Scripts\python scripts\remote_player.py --server %SERVER%
goto :eof
:nopython
echo Python 3.10+ is required: winget install Python.Python.3.12
