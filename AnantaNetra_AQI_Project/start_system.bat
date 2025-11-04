@echo off
REM AnantaNetra - Complete System Startup Script for Windows
REM This script starts both backend and frontend services

echo 🌍 Starting AnantaNetra - AI Environmental Monitoring System
echo ==================================================

REM Check if we're in the right directory
if not exist "README.md" (
    echo ❌ Error: Please run this script from the project root directory
    exit /b 1
)

REM Function to check if a port is in use
:check_port
netstat -an | findstr ":%1 " >nul 2>&1
exit /b

REM Start backend
echo 🚀 Starting Backend (FastAPI) on port 8000...
cd backend

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Start backend server in background
echo 🔄 Starting FastAPI server...
start "AnantaNetra Backend" cmd /k "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to start
echo ⏳ Waiting for backend to start...
timeout /t 10 /nobreak >nul

call :check_port 8000
if %errorlevel% equ 0 (
    echo ✅ Backend started successfully on http://localhost:8000
    echo 📚 API Documentation: http://localhost:8000/docs
) else (
    echo ❌ Failed to start backend
    exit /b 1
)

cd ..

REM Start frontend
echo 🎨 Starting Frontend (React + Vite) on port 5173...
cd AnantaNetra_AQI_Project\frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing Node.js dependencies...
    npm install
)

REM Start frontend development server
echo 🔄 Starting Vite development server...
start "AnantaNetra Frontend" cmd /k "npm run dev"

REM Wait for frontend to start
echo ⏳ Waiting for frontend to start...
timeout /t 15 /nobreak >nul

call :check_port 5173
if %errorlevel% equ 0 (
    echo ✅ Frontend started successfully on http://localhost:5173
) else (
    echo ❌ Failed to start frontend
    exit /b 1
)

cd ..\..

REM Show system status
echo.
echo 🎯 System Status:
echo ==================
echo 🔧 Backend API: http://localhost:8000
echo 📖 API Docs: http://localhost:8000/docs
echo 🌐 Frontend: http://localhost:5173
echo 💾 Demo Data: Available with fallback systems
echo.
echo 🔍 System Features:
echo   ✅ Real-time AQI monitoring
echo   ✅ 24-hour prediction forecasts
echo   ✅ AI-powered health advisories
echo   ✅ Interactive maps with city data
echo   ✅ Comprehensive error handling
echo   ✅ Responsive mobile design
echo.

REM Check for required tools
echo 🔍 Checking system requirements...

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    py --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Python is not installed. Please install Python 3.8+
        exit /b 1
    )
)

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 16+
    exit /b 1
)

REM Check npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not installed. Please install npm
    exit /b 1
)

echo ✅ All requirements satisfied

REM Check if ports are already in use
call :check_port 8000
if %errorlevel% equ 0 (
    echo ⚠️ Port 8000 is already in use. Please stop the existing service.
    exit /b 1
)

call :check_port 5173
if %errorlevel% equ 0 (
    echo ⚠️ Port 5173 is already in use. Please stop the existing service.
    exit /b 1
)

echo 🎮 Both services are starting in separate windows
echo 📊 Open http://localhost:5173 in your browser to access AnantaNetra
echo 🛑 Close the command windows to stop the services
echo.
echo Press any key to exit this startup script...
pause >nul
