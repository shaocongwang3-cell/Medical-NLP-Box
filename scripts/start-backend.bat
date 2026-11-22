@echo off
call D:\miniconda3\condabin\conda.bat activate medical-nlp
cd /d "%~dp0..\backend"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_ENDPOINT=https://hf-mirror.com
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
