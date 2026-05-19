@echo off
:: WEEKLY — COT backfill + Rollex sync + git push
:: Schedule: Friday ~21:00 (after CFTC release + Rollex master update)
:: For daily Rollex-only sync use run_daily.bat
set PYTHONIOENCODING=utf-8
set PYTHON=C:\Users\virat.arya\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe
set LOG=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Automator\run_log.txt
set REPO=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL
set INGEST_STATUS=ok
set GIT_STATUS=skipped

echo. >> "%LOG%"
echo ============================= >> "%LOG%"
echo Run started: %date% %time% >> "%LOG%"
echo ============================= >> "%LOG%"

:: Step 1 — Incremental COT ingest (all 6 commodities)
echo [1] Running cot_backfill.py... >> "%LOG%"
"%PYTHON%" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Code\cot_backfill.py" >> "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cot_backfill.py failed >> "%LOG%"
    set INGEST_STATUS=error
    goto notify
)

:: Step 1b — Sync Rollex and Roll Yield from master database folders
echo [1b] Syncing Rollex parquets... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Rollex\Database\rollex_*.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Database\Rollex\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Rollex sync had issues >> "%LOG%"

echo [1b] Syncing Roll Yield parquet... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Roll Yield\Database\roll_yield_data.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Database\RollYield\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Roll Yield sync had issues >> "%LOG%"

:: Step 2 — Push updated parquets to GitHub
echo [2] Pushing to GitHub... >> "%LOG%"
cd /d "%REPO%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cd to COT_ALL repo failed >> "%LOG%"
    set GIT_STATUS=failed
    goto notify
)
git add Database\cot_cit.parquet Database\cot_disagg_futopt.parquet Database\cot_disagg_fut.parquet Database\Rollex\rollex_*.parquet Database\RollYield\roll_yield_data.parquet >> "%LOG%" 2>&1
git diff --cached --quiet >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    git -c core.askpass= commit -m "Auto update: COT_ALL %date%" >> "%LOG%" 2>&1
    git -c core.askpass= pull --rebase --autostash >> "%LOG%" 2>&1
    git -c core.askpass= push >> "%LOG%" 2>&1
    if not errorlevel 1 (
        set GIT_STATUS=pushed
        echo Git push done. >> "%LOG%"
    ) else (
        set GIT_STATUS=failed
        echo ERROR: git push failed >> "%LOG%"
    )
) else (
    echo No changes to commit. >> "%LOG%"
    set GIT_STATUS=skipped
)

:notify
echo [3] Sending email notification... >> "%LOG%"
"%PYTHON%" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Automator\notify.py" %INGEST_STATUS% %GIT_STATUS% >> "%LOG%" 2>&1

echo Run finished: %date% %time% >> "%LOG%"
