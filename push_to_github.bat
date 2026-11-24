@echo off
REM First push of this project to an EMPTY GitHub repository.
cd /d "%~dp0"
set DEFAULT_URL=https://github.com/vijaysangi1432-cell/Amazon_Music_Clustering.git
set /p REPO_URL=GitHub repo URL [%DEFAULT_URL%]: 
if "%REPO_URL%"=="" set REPO_URL=%DEFAULT_URL%
git --version >nul 2>&1 || (echo Git is not installed. Install from https://git-scm.com & pause & exit /b 1)
if not exist .git git init
git add .
git commit -m "Amazon Music Clustering: pipeline, notebook, app, report and outputs"
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%
git push -u origin main
if errorlevel 1 (echo Push failed. Check the URL and that you are signed in to GitHub. & pause & exit /b 1)
echo.
echo Pushed to %REPO_URL%
pause
