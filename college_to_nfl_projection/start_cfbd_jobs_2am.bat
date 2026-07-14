@echo off
REM Restart CFBD data collection jobs at 2 AM EST
REM Run this via Windows Task Scheduler at 2:00 AM EST

echo ================================================================================
echo STARTING CFBD DATA COLLECTION JOBS
echo ================================================================================
echo.
echo Time: %date% %time%
echo.

cd /d "c:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\college_to_nfl_projection"

echo [1/2] Starting SP+ Historical Data Collection...
start "SP+ Historical Data" cmd /k "python src/fetch_historical_sp_plus_retry.py"
timeout /t 5 /nobreak > nul

echo [2/2] Starting College Career Data Collection...
start "College Career Data" cmd /k "python src/collect_full_college_careers_retry.py"
timeout /t 5 /nobreak > nul

echo.
echo ================================================================================
echo BOTH JOBS LAUNCHED
echo ================================================================================
echo.
echo Two terminal windows have been opened:
echo   1. SP+ Historical Data (retries every 10 minutes)
echo   2. College Career Data (retries every 30 minutes)
echo.
echo Both will run until all data is collected.
echo Close the windows to stop them.
echo.
pause
