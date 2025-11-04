#!/bin/bash

# AnantaNetra - Complete System Startup Script
# This script starts both backend and frontend services

echo "🌍 Starting AnantaNetra - AI Environmental Monitoring System"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    netstat -tuln | grep ":$1 " > /dev/null 2>&1
}

# Function to start backend
start_backend() {
    echo "🚀 Starting Backend (FastAPI) on port 8000..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating Python virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Start backend server
    echo "🔄 Starting FastAPI server..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    
    # Wait for backend to start
    echo "⏳ Waiting for backend to start..."
    sleep 10
    
    if check_port 8000; then
        echo "✅ Backend started successfully on http://localhost:8000"
        echo "📚 API Documentation: http://localhost:8000/docs"
    else
        echo "❌ Failed to start backend"
        exit 1
    fi
    
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting Frontend (React + Vite) on port 5173..."
    cd AnantaNetra_AQI_Project/frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing Node.js dependencies..."
        npm install
    fi
    
    # Start frontend development server
    echo "🔄 Starting Vite development server..."
    npm run dev &
    FRONTEND_PID=$!
    
    # Wait for frontend to start
    echo "⏳ Waiting for frontend to start..."
    sleep 15
    
    if check_port 5173; then
        echo "✅ Frontend started successfully on http://localhost:5173"
    else
        echo "❌ Failed to start frontend"
        exit 1
    fi
    
    cd ../..
}

# Function to show system status
show_status() {
    echo ""
    echo "🎯 System Status:"
    echo "=================="
    echo "🔧 Backend API: http://localhost:8000"
    echo "📖 API Docs: http://localhost:8000/docs"
    echo "🌐 Frontend: http://localhost:5173"
    echo "💾 Demo Data: Available with fallback systems"
    echo ""
    echo "🔍 System Features:"
    echo "  ✅ Real-time AQI monitoring"
    echo "  ✅ 24-hour prediction forecasts"
    echo "  ✅ AI-powered health advisories"
    echo "  ✅ Interactive maps with city data"
    echo "  ✅ Comprehensive error handling"
    echo "  ✅ Responsive mobile design"
    echo ""
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down AnantaNetra system..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend stopped"
    fi
    echo "👋 AnantaNetra system shutdown complete"
    exit 0
}

# Trap Ctrl+C and other signals
trap cleanup SIGINT SIGTERM

# Check for required tools
echo "🔍 Checking system requirements..."

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm"
    exit 1
fi

echo "✅ All requirements satisfied"

# Check if ports are already in use
if check_port 8000; then
    echo "⚠️  Port 8000 is already in use. Please stop the existing service."
    exit 1
fi

if check_port 5173; then
    echo "⚠️  Port 5173 is already in use. Please stop the existing service."
    exit 1
fi

# Start services
start_backend
start_frontend
show_status

# Keep script running
echo "🎮 Press Ctrl+C to stop all services"
echo "📊 Open http://localhost:5173 in your browser to access AnantaNetra"
echo ""

# Wait for user to stop
while true; do
    sleep 1
done
