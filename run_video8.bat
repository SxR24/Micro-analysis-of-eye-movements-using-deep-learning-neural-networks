@echo off
REM ======================================================================
REM run_video8.bat - torsion re-run for video 8, then merge, then compare.
REM
REM Run from the project root by double-clicking, or:  .\run_video8.bat
REM
REM Stage 1 is the slow one (~12 min). Stages 2 and 3 take seconds.
REM The baseline in data\video_8\baseline_prefix\ is never touched.
REM ======================================================================
setlocal
cd /d "%~dp0"

set VIDEO=data\raw\8.avi
set VDIR=data\video_8
set META=%VDIR%\frames\_frames_meta.json
set AOI=449,380,197

echo.
echo ============================================================
echo  STAGE 1/3  torsion tracking  (~15 min)
echo ============================================================
echo  Check these four lines appear before walking away:
echo    "RITnet coords -^> original space: ..."   (--meta took effect)
echo    "Feature gating: RITnet iris mask"        (lashes excluded)
echo    "Blink recovery window: 21 frames"        (420 ms window)
echo    "Capturing raw feature trajectories"      (npz export on)
echo.
python src/irisometry/ocular.py %VIDEO% --aoi %AOI% --ritnet %VDIR%\ritnet_8.csv --meta %META% --masks %VDIR%\masks --out %VDIR%
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  STAGE 2/3  merge RITnet + torsion
echo ============================================================
python src/irisometry/merge.py --ritnet %VDIR%\ritnet_8.csv --ocular %VDIR%\ocular_8.csv --meta %META% --out %VDIR%\combined_8.csv
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  STAGE 3/3  before/after comparison
echo ============================================================
echo --- vs the ORIGINAL pre-fix run ---
python src/irisometry/compare_runs.py --before %VDIR%\baseline_prefix\ocular_8_BEFORE.csv --after %VDIR%\ocular_8.csv --ritnet %VDIR%\ritnet_8.csv --out %VDIR%\comparison_vs_prefix
if errorlevel 1 goto :failed

echo.
echo --- vs the AOI-fixed run (isolates what mask gating alone bought) ---
python src/irisometry/compare_runs.py --before %VDIR%\baseline_aoifix\ocular_8_AOIFIX.csv --after %VDIR%\ocular_8.csv --ritnet %VDIR%\ritnet_8.csv --out %VDIR%\comparison_vs_aoifix
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  DONE
echo ============================================================
echo  Figures: %VDIR%\comparison_vs_prefix\comparison.png
echo           %VDIR%\comparison_vs_aoifix\comparison.png
echo  Replay : python src/review/live_view.py
echo  Review : streamlit run src/review/app.py
echo.
pause
exit /b 0

:failed
echo.
echo *** A STAGE FAILED - see the error above. Nothing downstream was run. ***
echo *** Your baseline in %VDIR%\baseline_prefix\ is untouched.            ***
echo.
pause
exit /b 1
