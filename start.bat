@echo off
echo =========================================
echo Starting AI Credit Prediction Server
echo =========================================

cd backend

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies (this may take a few minutes for PyTorch)...
pip install -r requirements.txt

echo Starting Flask Server...
python app.py

pause
