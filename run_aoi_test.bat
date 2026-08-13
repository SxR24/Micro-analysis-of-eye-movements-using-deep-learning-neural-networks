@echo off
REM ======================================================================
REM run_aoi_test.bat - single-variable test of the corrected AOI centre.
REM
REM The AOI in run_video8.bat (449,380,197) was seeded from the first 30
REM valid frames. get_aoi.py, estimating over the whole recording, gives
REM 454,407,191 -- 27 px lower. Since the AOI is LOCKED for the entire run,
REM a mis-centred circle clips the iris asymmetrically whenever the eye
REM looks away from wherever it happened to be pointing at the start.
REM
REM This writes to data\video_8\aoi_test\ and touches NOTHING in the main
REM output, so the current results stay intact for comparison.
REM
REM ~20 min. Segmentation is not re-run; the masks are unchanged.
REM ======================================================================
setlocal
cd /d "%~dp0"

set VIDEO=data\raw\8.avi
set VDIR=data\video_8
set META=%VDIR%\frames\_frames_meta.json
set OUT=%VDIR%\aoi_test

REM whole-recording estimate from get_aoi.py, instead of first-30-frames
set AOI=454,407,191

if not exist "%OUT%" mkdir "%OUT%"

echo.
echo ============================================================
echo  STAGE 1/3  torsion tracking with AOI %AOI%   (~15 min)
echo ============================================================
python src/irisometry/ocular.py %VIDEO% --aoi %AOI% --ritnet %VDIR%\ritnet_8.csv --meta %META% --masks %VDIR%\masks --out %OUT%
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  STAGE 2/3  merge
echo ============================================================
python src/irisometry/merge.py --ritnet %VDIR%\ritnet_8.csv --ocular %OUT%\ocular_8.csv --meta %META% --out %OUT%\combined_8.csv
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  STAGE 3/3  did the corrected AOI help?
echo ============================================================
echo  CURRENT  = corrected AOI %AOI%
echo  BASELINE = original AOI 449,380,197
echo.
echo  Judge on reliability and drift. A higher reliability means more
echo  of the torsion signal is real. Ignore jitter.
echo.
python src/analysis/reliability.py --features %OUT%\features_8.npz --baseline %VDIR%\features_8.npz
if errorlevel 1 goto :failed

echo.
python src/irisometry/compare_runs.py --before %VDIR%\ocular_8.csv --after %OUT%\ocular_8.csv --ritnet %VDIR%\ritnet_8.csv --out %OUT%\comparison
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  DONE - results in %OUT%
echo ============================================================
echo  If reliability went UP, adopt the new AOI: change the AOI line
echo  in run_video8.bat to %AOI% and re-run it.
echo  If it went DOWN or stayed flat, keep the original and record
echo  the negative result.
echo.
pause
exit /b 0

:failed
echo.
echo *** A STAGE FAILED - see the error above. ***
echo *** Your main results in %VDIR% are untouched.  ***
echo.
pause
exit /b 1
