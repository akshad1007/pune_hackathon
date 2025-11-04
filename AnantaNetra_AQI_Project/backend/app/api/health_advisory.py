from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.health_advisory import HealthAdvisoryResponse
from app.services.external_apis import external_api_service
from app.services.caching import cache_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/advisory", response_model=HealthAdvisoryResponse)
async def get_health_advisory(aqi: float = Query(..., ge=0, le=500)):
    """Get health advisory based on AQI value"""
    try:
        # Check cache
        cache_key = f"health_advisory:{int(aqi)}"
        cached_advisory = await cache_manager.get(cache_key)
        if cached_advisory:
            return HealthAdvisoryResponse(**cached_advisory)
        
        # Get AI-powered or static advisory
        advisory_data = await external_api_service.get_health_advisory_ai(aqi)
        
        # Create response
        advisory = HealthAdvisoryResponse(
            category=advisory_data.get("category", "Unknown"),
            message=advisory_data.get("message", "Consult local authorities for guidance."),
            precautions=advisory_data.get("precautions", ["Monitor air quality"]),
            risk_groups=advisory_data.get("risk_groups", ["General population"]),
            aqi_range=_get_aqi_range(aqi),
            health_effects=advisory_data.get("health_effects", "Potential health impacts based on AQI level")
        )
        
        # Cache for 2 hours
        await cache_manager.set(cache_key, advisory.dict(), ttl=7200)
        
        return advisory
        
    except Exception as e:
        logger.error(f"Health advisory error for AQI {aqi}: {e}")
        # Return fallback advisory
        return _get_fallback_advisory(aqi)

@router.get("/recommendations/{pincode}")
async def get_location_recommendations(pincode: str):
    """Get location-specific health recommendations"""
    try:
        if not pincode.isdigit() or len(pincode) != 6:
            raise HTTPException(status_code=400, detail="Invalid pincode format")
        
        # Get current AQI for the location
        from app.services.external_apis import external_api_service
        weather_data = await external_api_service.get_weather_data(pincode)
        current_aqi = weather_data.get('aqi', 150)
        
        # Get advisory
        advisory_data = await external_api_service.get_health_advisory_ai(current_aqi)
        
        # Add location-specific context
        location_context = _get_location_context(pincode)
        
        result = {
            "pincode": pincode,
            "current_aqi": current_aqi,
            "category": _get_aqi_category(current_aqi),
            "advisory": advisory_data,
            "location_context": location_context,
            "emergency_contacts": _get_emergency_contacts(pincode),
            "nearby_hospitals": _get_nearby_hospitals(pincode)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Location recommendations error for {pincode}: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@router.get("/alerts/{pincode}")
async def get_health_alerts(pincode: str):
    """Get active health alerts for a location"""
    try:
        if not pincode.isdigit() or len(pincode) != 6:
            raise HTTPException(status_code=400, detail="Invalid pincode format")
        
        # Get forecast to check for upcoming poor air quality
        from app.services.prediction import prediction_service
        forecast = await prediction_service.predict_aqi(pincode, hours_ahead=48)
        
        alerts = []
        
        # Check for severe air quality in forecast
        for prediction in forecast:
            if prediction.predicted_aqi > 300:
                alerts.append({
                    "type": "severe_air_quality",
                    "severity": "high",
                    "timestamp": prediction.timestamp,
                    "aqi": prediction.predicted_aqi,
                    "message": f"Very poor air quality expected at {prediction.timestamp[:10]}",
                    "actions": [
                        "Avoid all outdoor activities",
                        "Keep windows closed",
                        "Use air purifier",
                        "Consult doctor if experiencing symptoms"
                    ]
                })
            elif prediction.predicted_aqi > 200:
                alerts.append({
                    "type": "poor_air_quality",
                    "severity": "medium",
                    "timestamp": prediction.timestamp,
                    "aqi": prediction.predicted_aqi,
                    "message": f"Poor air quality expected at {prediction.timestamp[:10]}",
                    "actions": [
                        "Limit outdoor activities",
                        "Wear mask outdoors",
                        "Monitor symptoms"
                    ]
                })
        
        return {
            "pincode": pincode,
            "alerts_count": len(alerts),
            "alerts": alerts[:10],  # Limit to 10 most urgent
            "last_updated": _get_current_timestamp()
        }
        
    except Exception as e:
        logger.error(f"Health alerts error for {pincode}: {e}")
        return {"pincode": pincode, "alerts_count": 0, "alerts": [], "error": str(e)}

def _get_aqi_category(aqi_value: float) -> str:
    """Convert AQI value to category"""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Satisfactory"
    elif aqi_value <= 200:
        return "Moderate"
    elif aqi_value <= 300:
        return "Poor"
    elif aqi_value <= 400:
        return "Very Poor"
    else:
        return "Severe"

def _get_aqi_range(aqi_value: float) -> str:
    """Get AQI range description"""
    if aqi_value <= 50:
        return "0-50 (Good)"
    elif aqi_value <= 100:
        return "51-100 (Satisfactory)"
    elif aqi_value <= 200:
        return "101-200 (Moderate)"
    elif aqi_value <= 300:
        return "201-300 (Poor)"
    elif aqi_value <= 400:
        return "301-400 (Very Poor)"
    else:
        return "401-500 (Severe)"

def _get_fallback_advisory(aqi: float) -> HealthAdvisoryResponse:
    """Fallback advisory when AI/API fails"""
    if aqi <= 100:
        return HealthAdvisoryResponse(
            category="Good to Satisfactory",
            message="Air quality is acceptable for most people.",
            precautions=["No special precautions needed for general population"],
            risk_groups=["People with respiratory sensitivity"],
            aqi_range=_get_aqi_range(aqi),
            health_effects="No significant health effects expected"
        )
    elif aqi <= 200:
        return HealthAdvisoryResponse(
            category="Moderate",
            message="Air quality is unhealthy for sensitive groups.",
            precautions=[
                "Sensitive individuals should limit outdoor activities",
                "Consider wearing mask in high traffic areas",
                "Keep windows closed during peak hours"
            ],
            risk_groups=["Children", "Elderly", "People with heart/lung conditions"],
            aqi_range=_get_aqi_range(aqi),
            health_effects="Breathing discomfort for sensitive individuals"
        )
    else:
        return HealthAdvisoryResponse(
            category="Poor to Severe",
            message="Air quality is unhealthy for everyone.",
            precautions=[
                "Avoid outdoor activities",
                "Use N95 masks when outdoors",
                "Use air purifiers indoors",
                "Consult doctor if experiencing symptoms"
            ],
            risk_groups=["Entire population"],
            aqi_range=_get_aqi_range(aqi),
            health_effects="Serious health effects for all population groups"
        )

def _get_location_context(pincode: str) -> dict:
    """Get location-specific context"""
    city_contexts = {
        "400001": {"city": "Mumbai", "pollution_sources": ["Traffic", "Industrial", "Construction"], "seasonal_factors": ["Monsoon relief", "Winter inversion"]},
        "110001": {"city": "Delhi", "pollution_sources": ["Vehicular", "Stubble burning", "Industrial"], "seasonal_factors": ["Winter smog", "Dust storms"]},
        "560001": {"city": "Bangalore", "pollution_sources": ["Traffic", "Construction", "Industrial"], "seasonal_factors": ["Moderate year-round"]},
        "411001": {"city": "Pune", "pollution_sources": ["Automotive", "Industrial", "Dust"], "seasonal_factors": ["Seasonal variation"]}
    }
    
    return city_contexts.get(pincode, {
        "city": "Unknown",
        "pollution_sources": ["Traffic", "Industrial"],
        "seasonal_factors": ["Variable"]
    })

def _get_emergency_contacts(pincode: str) -> list:
    """Get emergency contacts for location"""
    return [
        {"service": "Emergency", "number": "108"},
        {"service": "Pollution Control Board", "number": "1800-180-1551"},
        {"service": "Medical Emergency", "number": "102"}
    ]

def _get_nearby_hospitals(pincode: str) -> list:
    """Get nearby hospitals (demo data)"""
    hospital_data = {
        "400001": [
            {"name": "JJ Hospital", "distance": "2.1 km", "speciality": "General, Emergency"},
            {"name": "GT Hospital", "distance": "3.5 km", "speciality": "Respiratory, Cardiology"}
        ],
        "110001": [
            {"name": "AIIMS Delhi", "distance": "5.2 km", "speciality": "Multi-specialty"},
            {"name": "RML Hospital", "distance": "2.8 km", "speciality": "Respiratory, Emergency"}
        ]
    }
    
    return hospital_data.get(pincode, [
        {"name": "Nearest Government Hospital", "distance": "Variable", "speciality": "General"}
    ])

def _get_current_timestamp() -> str:
    """Get current timestamp"""
    from datetime import datetime
    return datetime.now().isoformat()
