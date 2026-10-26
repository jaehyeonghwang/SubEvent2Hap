@echo off
rem Generation PC: serve the web UI to this PC and Tailscale peers only (other clients get 403).
cd /d "%~dp0"
set PYTHONWARNINGS=ignore
.venv\Scripts\python webui\server.py --host 0.0.0.0 %*
