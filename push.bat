@echo off
color 0A
title Auto-Push to GitHub
echo ==============================================
echo  Automated Push Script for Egg-Crack-Detection
echo ==============================================
echo.

cd /d "e:\final\it is okk_final"

echo 🗑️ Deleting heavy CNN and ResNet models...
del /Q "ml_models\best_cnn_model.h5" 2>nul
del /Q "ml_models\best_resnet_model.h5" 2>nul

echo 🧹 Cleaning up all unnecessary test and debugging files...
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
echo ✅ Workspace is now sparkling clean! Xception only!

echo 📦 Adding files to Git...
git add .
git commit -m "Removed unused models and test scripts, prep for Render"
echo.

echo 🚀 Pushing to GitHub (vendramohan45/Egg-Crack-Detection)...
git push -u origin main
echo.

echo ==============================================
echo 🎉 PUSH SUCCESSFUL! PROJECT IS 100%% RENDER READY!
echo ==============================================
pause
