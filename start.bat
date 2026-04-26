@echo off
cd /d "%~dp0"
title AI Master Agent - Jarvis
color 0A

echo ===================================================
echo   Starting Master AI Agent on Local Drive...
echo ===================================================

:: 1. إنشاء الفولدر المعزول
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Creating Isolated Environment "venv" inside this folder...
    python -m venv venv
)

:: 2. الدخول للبيئة المعزولة
echo [2/3] Activating Environment...
call venv\Scripts\activate.bat

:: 3. تسطيب المكتبات داخل فولدر venv فقط
echo [3/3] Checking and Installing Dependencies locally...
pip install gradio openai-whisper litellm openai python-docx pandas requests sounddevice torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 -q

echo ===================================================
echo   System Ready! Launching UI...
echo ===================================================

:: 4. تشغيل الأجنت
python master_agent3.py

pause