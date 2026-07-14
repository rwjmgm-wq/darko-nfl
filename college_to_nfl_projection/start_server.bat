@echo off
echo Starting QB Projection Visualizer...
echo.
echo Opening browser at http://localhost:8001
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
start http://localhost:8001/college_qb_explorer.html
python -m http.server 8001
