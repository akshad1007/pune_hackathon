# AnantaNetra - PowerShell Startup Script
# This script starts both backend and frontend services

Write-Host "🌍 Starting AnantaNetra - AI Environmental Monitoring System" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Yellow

# Check if we're in the right directory
if (-not (Test-Path "README.md")) {
    Write-Host "❌ Error: Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

Write-Host "🔍 Checking system requirements..." -ForegroundColor Cyan

# Check Python
try {
    $pythonVersion = python --version 2>$null
    if (-not $pythonVersion) {
        $pythonVersion = py --version 2>$null
    }
    if ($pythonVersion) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python is not installed. Please install Python 3.8+" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Python is not installed. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    if ($nodeVersion) {
        Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Node.js is not installed. Please install Node.js 16+" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Node.js is not installed. Please install Node.js 16+" -ForegroundColor Red
    exit 1
}

# Check npm
try {
    $npmVersion = npm --version 2>$null
    if ($npmVersion) {
        Write-Host "✅ npm found: $npmVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ npm is not installed. Please install npm" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ npm is not installed. Please install npm" -ForegroundColor Red
    exit 1
}

# Check if ports are already in use
if (Test-Port 8000) {
    Write-Host "⚠️ Port 8000 is already in use. Please stop the existing service." -ForegroundColor Yellow
    exit 1
}

if (Test-Port 5173) {
    Write-Host "⚠️ Port 5173 is already in use. Please stop the existing service." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ All requirements satisfied" -ForegroundColor Green
Write-Host ""

# Start Backend
Write-Host "🚀 Starting Backend (FastAPI) on port 8000..." -ForegroundColor Cyan
Set-Location "backend"

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "🔧 Activating Python virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Start backend server in new window
Write-Host "🔄 Starting FastAPI server..." -ForegroundColor Green
$backendJob = Start-Process powershell -ArgumentList "-NoExit", "-Command", "& 'venv\Scripts\Activate.ps1'; uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -PassThru

# Wait for backend to start
Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

if (Test-Port 8000) {
    Write-Host "✅ Backend started successfully on http://localhost:8000" -ForegroundColor Green
    Write-Host "📚 API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
} else {
    Write-Host "❌ Failed to start backend" -ForegroundColor Red
    exit 1
}

Set-Location ".."

# Start Frontend
Write-Host "🎨 Starting Frontend (React + Vite) on port 5173..." -ForegroundColor Cyan
Set-Location "AnantaNetra_AQI_Project\frontend"

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
}

# Start frontend development server in new window
Write-Host "🔄 Starting Vite development server..." -ForegroundColor Green
$frontendJob = Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev" -PassThru

# Wait for frontend to start
Write-Host "⏳ Waiting for frontend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

if (Test-Port 5173) {
    Write-Host "✅ Frontend started successfully on http://localhost:5173" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to start frontend" -ForegroundColor Red
    exit 1
}

Set-Location "..\..\"

# Show system status
Write-Host ""
Write-Host "🎯 System Status:" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Yellow
Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📖 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "🌐 Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "💾 Demo Data: Available with fallback systems" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔍 System Features:" -ForegroundColor Green
Write-Host "  ✅ Real-time AQI monitoring" -ForegroundColor White
Write-Host "  ✅ 24-hour prediction forecasts" -ForegroundColor White
Write-Host "  ✅ AI-powered health advisories" -ForegroundColor White
Write-Host "  ✅ Interactive maps with city data" -ForegroundColor White
Write-Host "  ✅ Comprehensive error handling" -ForegroundColor White
Write-Host "  ✅ Responsive mobile design" -ForegroundColor White
Write-Host ""
Write-Host "🎮 Both services are running in separate windows" -ForegroundColor Green
Write-Host "📊 Open http://localhost:5173 in your browser to access AnantaNetra" -ForegroundColor Yellow
Write-Host "🛑 Close the PowerShell windows to stop the services" -ForegroundColor Yellow
Write-Host ""
Write-Host "🏆 AnantaNetra is ready for your hackathon demo!" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press any key to exit this startup script..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
