@echo off
rem Laptop side: vibration to a SOUND OUTPUT (speaker / actuator amp) instead of the Quest,
rem so the narration can stay on the Windows default output (e.g. AirPods).
rem Usage: start_player_audio.bat ["device name part"] [server-url]
setlocal
cd /d "%~dp0"
set DEVICE=%~1
set SERVER=%~2
if "%SERVER%"=="" set SERVER=http://desktop-nst9a4u:7860
if not exist .venv-player\Scripts\python.exe (
  echo [setup] creating .venv-player ...
  py -3 -m venv .venv-player || python -m venv .venv-player || goto :nopython
  .venv-player\Scripts\python -m pip install -q --upgrade pip
)
.venv-player\Scripts\python -m pip install -q -r requirements-player.txt
if "%DEVICE%"=="" (
  .venv-player\Scripts\python scripts\remote_player.py --list-audio
  set /p DEVICE=Vibration output device ^(number or part of the name^):
)
echo Server: %SERVER%   Vibration device: %DEVICE%
.venv-player\Scripts\python scripts\remote_player.py --server %SERVER% --audio-device "%DEVICE%"
goto :eof
:nopython
echo Python 3.10+ is required: winget install Python.Python.3.12
