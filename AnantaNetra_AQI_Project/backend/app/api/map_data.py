from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from app.services.external_apis import external_api_service
from app.services.caching import cache_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/data")
async def get_map_data():
    """Get AQI data for map visualization"""
    try:
        cache_key = "map_data_all_cities"
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        
        # Demo cities with real coordinates
        demo_cities = [
            {"name": "Mumbai", "pincode": "400001", "lat": 19.0760, "lng": 72.8777},
            {"name": "Delhi", "pincode": "110001", "lat": 28.6139, "lng": 77.2090},
            {"name": "Pune", "pincode": "411001", "lat": 18.5204, "lng": 73.8567},
            {"name": "Bangalore", "pincode": "560001", "lat": 12.9716, "lng": 77.5946},
            {"name": "Chennai", "pincode": "600001", "lat": 13.0827, "lng": 80.2707},
            {"name": "Kolkata", "pincode": "700001", "lat": 22.5726, "lng": 88.3639},
            {"name": "Hyderabad", "pincode": "500001", "lat": 17.3850, "lng": 78.4867},
            {"name": "Ahmedabad", "pincode": "380001", "lat": 23.0225, "lng": 72.5714},
        ]
        
        map_data = []
        for city in demo_cities:
            try:
                # Get weather data for each city
                weather_data = await external_api_service.get_weather_data(city["pincode"])
                
                city_data = {
                    "city": city["name"],
                    "pincode": city["pincode"],
                    "latitude": city["lat"],
                    "longitude": city["lng"],
                    "aqi": weather_data.get("aqi", 150),
                    "category": _get_aqi_category(weather_data.get("aqi", 150)),
                    "pm25": weather_data.get("pm25", 60),
                    "pm10": weather_data.get("pm10", 90),
                    "temperature": weather_data.get("temperature", 28),
                    "humidity": weather_data.get("humidity", 65),
                    "wind_speed": weather_data.get("wind_speed", 12),
                    "last_updated": _get_current_timestamp()
                }
                map_data.append(city_data)
                
            except Exception as e:
                logger.warning(f"Failed to get data for {city['name']}: {e}")
                # Add fallback data
                map_data.append({
                    "city": city["name"],
                    "pincode": city["pincode"],
                    "latitude": city["lat"],
                    "longitude": city["lng"],
                    "aqi": 150,
                    "category": "Moderate",
                    "pm25": 60,
                    "pm10": 90,
                    "temperature": 28,
                    "humidity": 65,
                    "wind_speed": 12,
                    "last_updated": _get_current_timestamp(),
                    "source": "fallback"
                })
        
        result = {
            "cities": map_data,
            "total_cities": len(map_data),
            "last_updated": _get_current_timestamp(),
            "map_bounds": {
                "north": 35.0,
                "south": 8.0,
                "east": 97.0,
                "west": 68.0
            }
        }
        
        # Cache for 30 minutes
        await cache_manager.set(cache_key, result, ttl=1800)
        
        return result
        
    except Exception as e:
        logger.error(f"Map data error: {e}")
        # Return minimal fallback data
        return _get_fallback_map_data()

@router.get("/heatmap")
async def get_aqi_heatmap():
    """Get AQI heatmap data for visualization"""
    try:
        cache_key = "aqi_heatmap_data"
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        
        # Generate grid data for heatmap
        import random
        heatmap_data = []
        
        # Create grid across India
        lat_range = (8.0, 35.0)  # India's latitude range
        lng_range = (68.0, 97.0)  # India's longitude range
        
        grid_size = 20  # 20x20 grid
        lat_step = (lat_range[1] - lat_range[0]) / grid_size
        lng_step = (lng_range[1] - lng_range[0]) / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat = lat_range[0] + i * lat_step
                lng = lng_range[0] + j * lng_step
                
                # Generate realistic AQI values (higher in urban areas)
                base_aqi = random.randint(50, 300)
                
                # Add some geographic patterns
                if 28.0 <= lat <= 29.0 and 76.0 <= lng <= 78.0:  # Delhi region
                    base_aqi += 50
                elif 18.0 <= lat <= 20.0 and 72.0 <= lng <= 74.0:  # Mumbai region
                    base_aqi += 30
                
                aqi_value = max(10, min(500, base_aqi))
                
                heatmap_data.append({
                    "lat": round(lat, 4),
                    "lng": round(lng, 4),
                    "aqi": aqi_value,
                    "intensity": min(1.0, aqi_value / 300)  # Normalized intensity
                })
        
        result = {
            "heatmap_points": heatmap_data,
            "total_points": len(heatmap_data),
            "last_updated": _get_current_timestamp(),
            "legend": {
                "good": {"range": "0-50", "color": "#00e400"},
                "satisfactory": {"range": "51-100", "color": "#ffff00"},
                "moderate": {"range": "101-200", "color": "#ff7e00"},
                "poor": {"range": "201-300", "color": "#ff0000"},
                "very_poor": {"range": "301-400", "color": "#8f3f97"},
                "severe": {"range": "401-500", "color": "#7e0023"}
            }
        }
        
        # Cache for 1 hour
        await cache_manager.set(cache_key, result, ttl=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")

@router.get("/pollution-sources/{pincode}")
async def get_pollution_sources(pincode: str):
    """Get pollution sources analysis for a location"""
    try:
        if not pincode.isdigit() or len(pincode) != 6:
            raise HTTPException(status_code=400, detail="Invalid pincode format")
        
        cache_key = f"pollution_sources:{pincode}"
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            return cached_data
        
        # Demo pollution sources data based on location
        sources_data = _get_pollution_sources_data(pincode)
        
        result = {
            "pincode": pincode,
            "pollution_sources": sources_data,
            "last_updated": _get_current_timestamp()
        }
        
        # Cache for 4 hours
        await cache_manager.set(cache_key, result, ttl=14400)
        
        return result
        
    except Exception as e:
        logger.error(f"Pollution sources error for {pincode}: {e}")
        raise HTTPException(status_code=500, detail=f"Pollution sources analysis failed: {str(e)}")

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
    """Get current timestamp"""
    from datetime import datetime
    return datetime.now().isoformat()

def _get_fallback_map_data() -> Dict[str, Any]:
    """Fallback map data when APIs fail"""
    return {
        "cities": [
            {
                "city": "Mumbai",
                "pincode": "400001",
                "latitude": 19.0760,
                "longitude": 72.8777,
                "aqi": 150,
                "category": "Moderate",
                "pm25": 60,
                "pm10": 90,
                "temperature": 28,
                "humidity": 65,
                "wind_speed": 12,
                "source": "fallback"
            },
            {
                "city": "Delhi",
                "pincode": "110001",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "aqi": 200,
                "category": "Poor",
                "pm25": 80,
                "pm10": 120,
                "temperature": 25,
                "humidity": 70,
                "wind_speed": 8,
                "source": "fallback"
            }
        ],
        "total_cities": 2,
        "last_updated": _get_current_timestamp(),
        "status": "fallback_mode"
    }

def _get_pollution_sources_data(pincode: str) -> List[Dict[str, Any]]:
    """Get pollution sources data for a pincode"""
    
    # City-specific pollution sources
    sources_by_city = {
        "400001": [  # Mumbai
            {"source": "Vehicular Traffic", "contribution": 35, "trend": "increasing"},
            {"source": "Industrial Emissions", "contribution": 25, "trend": "stable"},
            {"source": "Construction Activities", "contribution": 20, "trend": "increasing"},
            {"source": "Marine Emissions", "contribution": 15, "trend": "stable"},
            {"source": "Waste Burning", "contribution": 5, "trend": "decreasing"}
        ],
        "110001": [  # Delhi
            {"source": "Vehicular Traffic", "contribution": 40, "trend": "increasing"},
            {"source": "Stubble Burning", "contribution": 25, "trend": "seasonal"},
            {"source": "Industrial Emissions", "contribution": 20, "trend": "stable"},
            {"source": "Dust (Construction/Road)", "contribution": 10, "trend": "increasing"},
            {"source": "Waste Burning", "contribution": 5, "trend": "stable"}
        ],
        "560001": [  # Bangalore
            {"source": "Vehicular Traffic", "contribution": 45, "trend": "increasing"},
            {"source": "Construction Activities", "contribution": 25, "trend": "increasing"},
            {"source": "Industrial Emissions", "contribution": 20, "trend": "stable"},
            {"source": "Road Dust", "contribution": 8, "trend": "stable"},
            {"source": "Waste Burning", "contribution": 2, "trend": "decreasing"}
        ]
    }
    
    return sources_by_city.get(pincode, [
        {"source": "Vehicular Traffic", "contribution": 40, "trend": "stable"},
        {"source": "Industrial Emissions", "contribution": 30, "trend": "stable"},
        {"source": "Construction Activities", "contribution": 20, "trend": "stable"},
        {"source": "Other Sources", "contribution": 10, "trend": "stable"}
    ])
