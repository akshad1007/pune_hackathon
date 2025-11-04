from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

async def handle_api_error(error: Exception, message: str) -> JSONResponse:
    """Handle API errors with fallback responses"""
    logger.error(f"API Error: {message} - {str(error)}")
    
    # Return structured error response with fallback data
    fallback_data = get_fallback_data(message)
    
    return JSONResponse(
        status_code=200,  # Return 200 with fallback data
        content={
            "status": "fallback",
            "message": f"{message} - Using cached/demo data",
            "data": fallback_data,
            "error_details": str(error)
        }
    )

def get_fallback_data(context: str) -> Dict[str, Any]:
    """Get appropriate fallback data based on context"""
    
    if "aqi" in context.lower():
        return {
            "aqi": 150,
            "category": "Moderate",
            "pincode": "400001",
            "timestamp": "2025-09-01T12:00:00Z",
            "source": "fallback",
            "pm25": 90,
            "pm10": 120,
            "temperature": 28,
            "humidity": 65,
            "wind_speed": 12
        }
    
    elif "forecast" in context.lower():
        from datetime import datetime, timedelta
        base_time = datetime.now()
        
        return [
            {
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "predicted_aqi": 120 + (i * 5),
                "category": "Moderate",
                "confidence_lower": 100,
                "confidence_upper": 180
            }
            for i in range(1, 25)
        ]
    
    elif "health" in context.lower():
        return {
            "category": "Moderate",
            "message": "Sensitive individuals should limit outdoor activities.",
            "precautions": [
                "Wear mask when going outdoors",
                "Keep windows closed",
                "Use air purifier indoors",
                "Avoid outdoor exercise"
            ],
            "risk_groups": ["Children", "Elderly", "People with respiratory conditions"]
        }
    
    elif "map" in context.lower():
        return [
            {
                "city": "Mumbai",
                "pincode": "400001",
                "latitude": 19.0760,
                "longitude": 72.8777,
                "aqi": 150,
                "category": "Moderate"
            },
            {
                "city": "Delhi",
                "pincode": "110001", 
                "latitude": 28.6139,
                "longitude": 77.2090,
                "aqi": 200,
                "category": "Poor"
            }
        ]
    
    else:
        return {
            "status": "Service temporarily unavailable",
            "fallback": True
        }
