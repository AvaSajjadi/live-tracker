@echo off
rem Run MARS pose on a video on THIS computer (video in, keypoints.parquet and stretched.mp4 out).
rem The models and the Python environment live in %USERPROFILE%\mars_local (outside the project, not in Git).
rem
rem   scripts\mars_local.cmd my_video.mp4
rem   scripts\mars_local.cmd my_video.mp4 --corners 0.23,0.22,0.81,0.22,0.81,0.65,0.23,0.65 --max-frames 300
rem
rem Any option of mars_pose_offline.py can be added (--out, --start-frame, --max-frames, --batch, --session ...).
rem This laptop does about 1 frame per second, so use a short clip first (for example --max-frames 300).
set "MARS_HOME=%USERPROFILE%\mars_local"
if not exist "%MARS_HOME%\env\Scripts\python.exe" (
  echo MARS is not set up: %MARS_HOME%\env is missing. See docs\camera.md, section 8.
  exit /b 1
)
"%MARS_HOME%\env\Scripts\python.exe" "%~dp0mars_pose_offline.py" --models-dir "%MARS_HOME%\models" --out keypoints.parquet --save-video stretched.mp4 --batch 4 %*
