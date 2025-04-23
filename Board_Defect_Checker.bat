@echo off
REM Change directory to your project folder (adjust the path as needed)
cd /d "D:\Working\BoardDefectChecker"

REM Activate the virtual environment (using the batch activation script)
call .venv\Scripts\activate

REM Run your main.py script
python main.py

REM Optional: Pause to view output before the window closes
pause
