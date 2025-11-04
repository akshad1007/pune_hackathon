from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from app.schemas.aqi import AQIResponse
from app.services.external_apis import external_api_service
from app.services.caching import cache_manager
from app.utils.error_handling import handle_api_error

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{pincode}", response_model=AQIResponse)
async def get_aqi_by_pincode(pincode: str):
    """Get current AQI for a specific pincode"""
    try:
        # Validate pincode format
        if not pincode.isdigit() or len(pincode) != 6:
            raise HTTPException(status_code=400, detail="Invalid pincode format")
        
        # Check cache first
        cache_key = f"aqi:{pincode}"
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"Returning cached AQI data for {pincode}")
            return cached_data
        
        # Fetch live data
        weather_data = await external_api_service.get_weather_data(pincode)
        geocoding_data = await external_api_service.get_geocoding_data(pincode)
        
        # Create AQI response
        aqi_data = AQIResponse(
            aqi=weather_data.get('aqi', 150),
            category=_get_aqi_category(weather_data.get('aqi', 150)),
            pincode=pincode,
            timestamp=_get_current_timestamp(),
            source=weather_data.get('source', 'api'),
            pm25=weather_data.get('pm25'),
            pm10=weather_data.get('pm10'),
            temperature=weather_data.get('temperature'),
            humidity=weather_data.get('humidity'),
            wind_speed=weather_data.get('wind_speed')
        )
        
        # Cache the result for 30 minutes
        await cache_manager.set(cache_key, aqi_data.dict(), ttl=1800)
        
        logger.info(f"Returning fresh AQI data for {pincode}: {aqi_data.aqi}")
        return aqi_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching AQI for pincode {pincode}: {e}")
        return await handle_api_error(e, f"Failed to fetch AQI for {pincode}")

@router.get("/district/{district_name}")
async def get_aqi_by_district(district_name: str):
    """Get current AQI for a district"""
    try:
        # This would typically query multiple pincodes in the district
        # For demo, return district-level aggregate data
        
        cache_key = f"district_aqi:{district_name.lower()}"
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        
        # Demo district data
        district_data = {
            "district": district_name,
            "average_aqi": 165,
            "category": "Moderate", 
            "stations": [
                {"pincode": "400001", "aqi": 150, "location": "Central"},
                {"pincode": "400002", "aqi": 180, "location": "Suburban"}
            ],
            "timestamp": _get_current_timestamp(),
            "source": "district_aggregate"
        }
        
        await cache_manager.set(cache_key, district_data, ttl=3600)
        return district_data
        
    except Exception as e:
        return await handle_api_error(e, f"Failed to fetch AQI for district {district_name}")

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

def _get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    from datetime import datetime
    return datetime.now().isoformat()
