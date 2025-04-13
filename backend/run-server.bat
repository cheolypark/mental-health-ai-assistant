@echo off
REM Activate your virtual environment
call conda activate my-manim-environment

REM Run your FastAPI server
uvicorn server-llama:app --host 0.0.0.0 --port 8080
