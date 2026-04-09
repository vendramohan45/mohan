@echo off
color 0E
title Hugging Face Deployment Push
echo ==============================================
echo  🚀 EggDetect AI: Pushing to GitHub for HF
echo ==============================================
echo.

cd /d "e:\final\it is okk_final"

echo 📦 Adding files to Git...
git add .
git commit -m "Add Docker configuration and Hugging Face compatibility settings"

echo 🚀 Pushing to GitHub (origin main)...
git push origin main

echo.
echo ==============================================
echo 🎉 PUSH COMPLETE!
echo.
echo Your changes are now on GitHub.
echo Now go to your Hugging Face Space settings and 
echo click 'Sync from GitHub' or wait for auto-deploy.
echo ==============================================
pause
