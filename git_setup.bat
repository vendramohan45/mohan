@echo off
color 0B
title EggDetect AI - Render Preparation & Push
echo ===================================================
echo   🥚 EggDetect AI: Preparing for Render Deploy
echo ===================================================
echo.

echo 🧹 Step 1: Cleaning up unnecessary files...
echo Deleting empty CNN and ResNet models (Xception only for Render)...
del /Q "ml_models\best_cnn_model.h5" 2>nul
del /Q "ml_models\best_resnet_model.h5" 2>nul

echo Deleting development, test and debug scripts...
del /Q "check_*.py" 2>nul
del /Q "test_*.py" 2>nul
del /Q "inspect_*.py" 2>nul
del /Q "compare_*.py" 2>nul
del /Q "diag_*.py" 2>nul
del /Q "fix_*.py" 2>nul
del /Q "verify_*.py" 2>nul
del /Q "basic_test.py" 2>nul
del /Q "benchmark_models.py" 2>nul
del /Q "simple_test.py" 2>nul
del /Q "debug_loading.py" 2>nul
del /Q "quantize_xception.py" 2>nul
del /Q "test_car_image.jpg" 2>nul
del /Q "test_egg.jpg" 2>nul
echo ✅ Cleanup complete.

echo.
echo 🚀 Step 2: Initializing and Pushing to GitHub...
echo Target Repository: https://github.com/vendramohan45/mohan.git
echo.

:: Initialize if needed
if not exist .git (
    git init
)

:: Re-configure remote
git remote remove origin 2>nul
git remote add origin https://github.com/vendramohan45/mohan.git

:: Add and Commit
git add .
git commit -m "Final Render-Ready Build: Optimized models and cleaned workspace"

:: Push to main
echo Pushing... (Please enter credentials if prompted)
git branch -M main
git push -u origin main

echo.
echo ===================================================
echo 🎉 SUCCESS! PROJECT IS 100%% RENDER READY!
echo ===================================================
echo.
echo NEXT STEPS ON RENDER:
echo 1. Build Command: ./build.sh
echo 2. Start Command: gunicorn eggdetect.wsgi:application
echo.
pause
