from fastapi import APIRouter, HTTPException, Query
from typing import List
import logging

from app.schemas.forecast import ForecastResponse
from app.services.prediction import prediction_service
from app.services.caching import cache_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{pincode}", response_model=List[ForecastResponse])
async def get_aqi_forecast(
    pincode: str, 
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    confidence_interval: bool = True
):
    """Get AQI forecast for next N hours"""
    try:
        # Validate pincode
        if not pincode.isdigit() or len(pincode) != 6:
            raise HTTPException(status_code=400, detail="Invalid pincode format")
        
        # Check cache
        cache_key = f"forecast:{pincode}:{hours}:{confidence_interval}"
        cached_forecast = await cache_manager.get(cache_key)
        if cached_forecast:
            logger.info(f"Returning cached forecast for {pincode}")
            # Convert dict back to ForecastResponse objects
            return [ForecastResponse(**item) for item in cached_forecast]
        
        # Generate predictions
        logger.info(f"Generating {hours}h forecast for pincode {pincode}")
        forecast = await prediction_service.predict_aqi(
            pincode=pincode,
            hours_ahead=hours,
            include_confidence=confidence_interval
        )
        
        # Cache results for 1 hour
        forecast_dicts = [item.dict() for item in forecast]
        await cache_manager.set(cache_key, forecast_dicts, ttl=3600)
        
        logger.info(f"Generated forecast with {len(forecast)} predictions")
        return forecast
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error for {pincode}: {e}")
        # Return fallback forecast
        return await prediction_service._generate_fallback_forecast(pincode, hours)

@router.get("/trend/{pincode}")
async def get_aqi_trend(pincode: str, days: int = Query(7, ge=1, le=30)):
    """Get historical AQI trend for analysis"""
    try:
        if not pincode.isdigit() or len(pincode) != 6:
            raise HTTPException(status_code=400, detail="Invalid pincode format")
        
        cache_key = f"trend:{pincode}:{days}"
        cached_trend = await cache_manager.get(cache_key)
        if cached_trend:
            return cached_trend
        
        # Generate demo trend data
        from datetime import datetime, timedelta
        import random
        
        trend_data = []
        base_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 24):  # Hourly data
            timestamp = base_time + timedelta(hours=i)
            
            # Simulate realistic AQI pattern
            base_aqi = 120 + random.randint(-30, 50)
            hour_factor = 1.2 if 7 <= timestamp.hour <= 20 else 0.8
            day_factor = 1.1 if timestamp.weekday() < 5 else 0.9  # Weekday vs weekend
            
            aqi = max(10, min(500, base_aqi * hour_factor * day_factor))
            
            trend_data.append({
                "timestamp": timestamp.isoformat(),
                "aqi": round(aqi, 1),
                "category": _get_aqi_category(aqi)
            })
        
        result = {
            "pincode": pincode,
            "period_days": days,
            "data_points": len(trend_data),
            "average_aqi": sum(item["aqi"] for item in trend_data) / len(trend_data),
            "trend_data": trend_data
        }
        
        await cache_manager.set(cache_key, result, ttl=7200)  # 2 hours
        return result
        
    except Exception as e:
        logger.error(f"Trend error for {pincode}: {e}")
        raise HTTPException(status_code=500, detail=f"Trend generation failed: {str(e)}")

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
