@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =============================================================
REM  Multi-run script: runs 1-5 with different negative sampling seeds.
REM
REM  Current defaults:
REM    - IDs are ENSG gene IDs from the start.
REM    - Negatives use DeepND mapped negatives first, then SFARI-filtered
REM      random fill to 1:1 positives:negatives.
REM    - Deep-MOTIFs PU class prior defaults to pi=0.06.
REM
REM  Note:
REM    lg.py and lstm.py are not present in this workspace, so LightGBM and
REM    standalone LSTM are not run here.
REM    Deep-MOTIFs uses the final Bayesian-guided PU defaults selected by
REM    sensitivity analysis: pi=0.06, delta=0.05, weight floor=0.10,
REM    rank calibration, RRF fusion, and no PPR.
REM
REM  Usage:
REM    cd C:\Users\wdi16\OneDrive\Desktop\shengtong\revision\data
REM    run_multi.bat
REM    run_multi.bat 1   REM only continue/run run_1
REM
REM  Resume behavior:
REM    Existing completed outputs are skipped. Delete an output directory if
REM    you want to force rerun a specific algorithm.
REM =============================================================

set "PROJECT=C:\Users\wdi16\OneDrive\Desktop\shengtong\revision\data"
set "PYTHON=C:\Users\wdi16\anaconda3\envs\tor\python.exe"
set "TARGET_NEG_RATIO=1.0"

cd /d "%PROJECT%" || exit /b 1

if "%~1"=="1" (
  call :run_all 1 42 || goto failed
  echo.
  echo [DONE] run_1 completed.
  exit /b 0
)

call :run_all 1 42    || goto failed
call :run_all 2 123   || goto failed
call :run_all 3 456   || goto failed
call :run_all 4 789   || goto failed
call :run_all 5 1024  || goto failed

echo.
echo ============================================================
echo  Post-processing run_1..run_5
echo ============================================================

%PYTHON% average_runs.py
call :check "average_runs.py" || goto failed

%PYTHON% add_mcc_auc_metrics.py
call :check "add_mcc_auc_metrics.py" || goto failed

echo.
echo ============================================================
echo  All runs completed.
echo  Results: run_1\  run_2\  run_3\  run_4\  run_5\
echo  Mean:    mean_run\
echo ============================================================
exit /b 0


:run_all
set "RUN_ID=%~1"
set "NEG_SEED=%~2"
set "RUN_DIR=run_%RUN_ID%"
set "LABELS_DIR=%PROJECT%\%RUN_DIR%\forecasd_outputs"
set "FORECASD_RERAN=0"

echo.
echo ============================================================
echo  [Run %RUN_ID% / 5] neg-random-state=%NEG_SEED%
echo ============================================================

mkdir "%PROJECT%\%RUN_DIR%" 2>nul

call :run_forecasd_if_needed "Run %RUN_ID% forecasd" "%PROJECT%\%RUN_DIR%\forecasd_outputs\all_labels_used.csv" "%PROJECT%\%RUN_DIR%\forecasd_outputs\neg_selection_report.json" forecasd.py --project-root "%PROJECT%" --neg-random-state %NEG_SEED% --target-neg-ratio %TARGET_NEG_RATIO% --output-dir "%RUN_DIR%\forecasd_outputs"
if errorlevel 1 exit /b 1

call :run_xgb || exit /b 1
call :run_rf || exit /b 1
call :run_sv || exit /b 1
call :run_deepgbm || exit /b 1
call :run_sai || exit /b 1
call :run_tab || exit /b 1
call :run_ftt || exit /b 1
call :run_cnn || exit /b 1
call :run_gcn || exit /b 1
call :run_deep_motifs || exit /b 1

echo [Run %RUN_ID% done]
exit /b 0


:run_forecasd_if_needed
set "STEP_NAME=%~1"
set "DONE_FILE=%~2"
set "REPORT_FILE=%~3"
shift
shift
shift
if exist "%DONE_FILE%" if exist "%REPORT_FILE%" (
  "%PYTHON%" -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if abs(float(r.get('achieved_neg_ratio', -999))-float(sys.argv[2])) < 1e-9 else 1)" "%REPORT_FILE%" "%TARGET_NEG_RATIO%"
  if not errorlevel 1 (
    echo [SKIP] %STEP_NAME% already completed with ratio %TARGET_NEG_RATIO%: %DONE_FILE%
    exit /b 0
  )
  echo [RERUN] %STEP_NAME% exists but ratio is not %TARGET_NEG_RATIO%.
)
echo [RUN] %STEP_NAME%
set "RUN_ARGS="
:collect_forecasd_args
if "%~1"=="" goto execute_forecasd_args
set "RUN_ARGS=!RUN_ARGS! ^"%~1^""
shift
goto collect_forecasd_args
:execute_forecasd_args
%PYTHON% !RUN_ARGS!
call :check "%STEP_NAME%"
if not errorlevel 1 set "FORECASD_RERAN=1"
exit /b %errorlevel%


:run_xgb
set "STEP_NAME=Run %RUN_ID% xgb"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\xgb_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% xgb.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\xgb_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_rf
set "STEP_NAME=Run %RUN_ID% rf"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\rf_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% rf.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\rf_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_sv
set "STEP_NAME=Run %RUN_ID% sv"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\sv_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% sv.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\sv_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_deepgbm
set "STEP_NAME=Run %RUN_ID% deepgbm"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\deepgbm_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% deepgbm.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\deepgbm_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_sai
set "STEP_NAME=Run %RUN_ID% sai"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\sai_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% sai.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\sai_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_tab
set "STEP_NAME=Run %RUN_ID% tab"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\tab_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% tab.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\tab_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_ftt
set "STEP_NAME=Run %RUN_ID% ftt"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\ftt_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% ftt.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\ftt_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_cnn
set "STEP_NAME=Run %RUN_ID% cnn"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\cnn_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% cnn.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\cnn_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_gcn
set "STEP_NAME=Run %RUN_ID% gcn"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\gcn_outputs\cv_metrics_summary.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% gcn.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\gcn_outputs"
call :check "%STEP_NAME%"
exit /b %errorlevel%

:run_deep_motifs
set "STEP_NAME=Run %RUN_ID% deep_motifs"
set "DONE_FILE=%PROJECT%\%RUN_DIR%\deep_motifs_outputs\cv_metrics_summary_global_threshold.csv"
if exist "%DONE_FILE%" if not "%FORECASD_RERAN%"=="1" (echo [SKIP] %STEP_NAME% already completed: %DONE_FILE%& exit /b 0)
if exist "%DONE_FILE%" echo [RERUN] %STEP_NAME% because labels were regenerated.
%PYTHON% deep_motifs.py --project-root "%PROJECT%" --labels-dir "%LABELS_DIR%" --output-dir "%RUN_DIR%\deep_motifs_outputs" --prior-model lstm --fusion-mode rrf --ppr-alpha 1.0 --pu-class-prior 0.06 --prior-uncertainty-delta 0.05 --prior-weight-floor 0.10 --prior-guided-calibration rank
call :check "%STEP_NAME%"
exit /b %errorlevel%


:check
if errorlevel 1 (
  echo.
  echo [FAILED] %~1
  exit /b 1
)
echo [OK] %~1
exit /b 0


:failed
echo.
echo ============================================================
echo  Stopped because one step failed.
echo ============================================================
exit /b 1
