@echo off
set PYTHONIOENCODING=utf-8
set PYTHON=C:\Users\virat.arya\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe
set LOG=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Automator\run_daily_log.txt
set REPO=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL
set GIT_STATUS=skipped

echo. >> "%LOG%"
echo ============================= >> "%LOG%"
echo Daily run started: %date% %time% >> "%LOG%"
echo ============================= >> "%LOG%"

:: Step 1 — Sync Rollex parquets from master database
echo [1] Syncing Rollex parquets... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Rollex\Database\rollex_*.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Database\Rollex\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Rollex sync had issues >> "%LOG%"

:: Step 2 — Sync Roll Yield parquet from master database
echo [2] Syncing Roll Yield parquet... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Roll Yield\Database\roll_yield_data.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Database\RollYield\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Roll Yield sync had issues >> "%LOG%"

:: Step 3 — Push updated parquets to GitHub
echo [3] Pushing to GitHub... >> "%LOG%"
cd /d "%REPO%"
git add Database\Rollex\rollex_*.parquet Database\RollYield\roll_yield_data.parquet >> "%LOG%" 2>&1
git diff --cached --quiet
if %ERRORLEVEL% NEQ 0 (
    git -c core.askpass= commit -m "Daily Rollex sync: %date%" >> "%LOG%" 2>&1
    git -c core.askpass= pull --rebase --autostash >> "%LOG%" 2>&1
    git -c core.askpass= push >> "%LOG%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        set GIT_STATUS=failed
        echo ERROR: git push failed >> "%LOG%"
    ) else (
        set GIT_STATUS=pushed
        echo Git push done. >> "%LOG%"
    )
) else (
    echo No changes to commit. >> "%LOG%"
    set GIT_STATUS=skipped
)

:: Step 4 — Email notification
echo [4] Sending notification... >> "%LOG%"
"%PYTHON%" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Automator\notify.py" ok %GIT_STATUS% >> "%LOG%" 2>&1

echo Daily run finished: %date% %time% >> "%LOG%"
