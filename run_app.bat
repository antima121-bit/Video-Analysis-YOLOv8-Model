@echo off
echo Starting Video Analysis Application...

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Install requirements
echo Installing requirements...
py -m pip install -r requirements.txt

:: Create necessary directories
echo Creating directories...
mkdir src\uploads 2>nul
mkdir src\static\results 2>nul

:: Start the Flask server
echo Starting Flask server...
py api/web_demo.py

:: Open the browser
timeout /t 2
start http://127.0.0.1:5000

echo Application is running!
echo Press Ctrl+C to stop the server 