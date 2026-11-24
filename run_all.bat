@echo off
REM Amazon Music Clustering - install, run pipeline, execute notebook
cd /d "%~dp0"
echo [1/3] Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo [2/3] Running clustering pipeline (about 2 minutes)...
python -m src.run_pipeline
if errorlevel 1 goto :error
echo.
echo [3/3] Executing notebook...
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks\Amazon_Music_Clustering.ipynb
if errorlevel 1 goto :error
echo.
echo ==========================================================
echo  Done. Outputs: outputs\amazon_music_clustered.csv
echo  Start the app with:  streamlit run app\app.py
echo ==========================================================
pause
exit /b 0
:error
echo.
echo Something failed above. Copy the red error text and share it.
pause
exit /b 1
