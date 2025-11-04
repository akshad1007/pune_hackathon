import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

from app.utils.config import settings
from app.services.caching import cache_manager

logger = logging.getLogger(__name__)

class ExternalAPIService:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
    
    async def get_weather_data(self, pincode: str) -> Dict[str, Any]:
        """Get weather data from external APIs"""
        try:
            # Try WeatherAPI first
            if settings.weather_api_key and not settings.demo_mode:
                url = f"http://api.weatherapi.com/v1/current.json"
                params = {
                    "key": settings.weather_api_key,
                    "q": pincode,
                    "aqi": "yes"
                }
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_weather_api_response(data)
            
            # Fallback to OpenWeatherMap
            if settings.openweather_api_key and not settings.demo_mode:
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    "appid": settings.openweather_api_key,
                    "q": pincode,
                    "units": "metric"
                }
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_openweather_response(data)
        
        except Exception as e:
            logger.error(f"Weather API error: {e}")
        
        # Return fallback data
        return self._get_fallback_weather_data(pincode)
    
    async def get_geocoding_data(self, pincode: str) -> Dict[str, Any]:
        """Get geocoding data for pincode"""
        try:
            if settings.opencage_api_key and not settings.demo_mode:
                url = f"https://api.opencagedata.com/geocode/v1/json"
                params = {
                    "key": settings.opencage_api_key,
                    "q": pincode + ", India",
                    "limit": 1
                }
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data["results"]:
                        result = data["results"][0]
                        return {
                            "latitude": result["geometry"]["lat"],
                            "longitude": result["geometry"]["lng"],
                            "city": result["components"].get("city", "Unknown"),
                            "state": result["components"].get("state", "Unknown")
                        }
        
        except Exception as e:
            logger.error(f"Geocoding API error: {e}")
        
        # Return fallback coordinates
        return self._get_fallback_geocoding_data(pincode)
    
    async def get_health_advisory_ai(self, aqi_value: float) -> Dict[str, Any]:
        """Get AI-powered health advisory"""
        try:
            if settings.gemini_api_key and not settings.demo_mode:
                import google.generativeai as genai
                
                genai.configure(api_key=settings.gemini_api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                prompt = f"""
                Given an AQI value of {aqi_value}, provide specific health advisory for Indian citizens.
                Include:
                1. Health impact category
                2. Specific precautions 
                3. Risk groups
                4. Recommended actions
                
                Keep response concise and actionable. Format as JSON with keys:
                category, message, precautions (array), risk_groups (array), health_effects
                """
                
                response = model.generate_content(prompt)
                # Parse AI response
                return self._parse_ai_response(response.text, aqi_value)
        
        except Exception as e:
            logger.error(f"AI Advisory error: {e}")
        
        # Return static health advisory
        return self._get_static_health_advisory(aqi_value)
    
    def _parse_weather_api_response(self, data: Dict) -> Dict[str, Any]:
        """Parse WeatherAPI response"""
        current = data.get("current", {})
        air_quality = current.get("air_quality", {})
        
        return {
            "temperature": current.get("temp_c", 25),
            "humidity": current.get("humidity", 60),
            "wind_speed": current.get("wind_kph", 10),
            "pressure": current.get("pressure_mb", 1013),
            "pm25": air_quality.get("pm2_5", 50),
            "pm10": air_quality.get("pm10", 80),
            "aqi": self._calculate_aqi_from_pm(air_quality.get("pm2_5", 50)),
            "source": "WeatherAPI"
        }
    
    def _parse_openweather_response(self, data: Dict) -> Dict[str, Any]:
        """Parse OpenWeatherMap response"""
        main = data.get("main", {})
        wind = data.get("wind", {})
        
        return {
            "temperature": main.get("temp", 25),
            "humidity": main.get("humidity", 60),
            "wind_speed": wind.get("speed", 3) * 3.6,  # m/s to km/h
            "pressure": main.get("pressure", 1013),
            "pm25": 50,  # OpenWeather doesn't provide air quality in free tier
            "pm10": 80,
            "aqi": 150,  # Default moderate
            "source": "OpenWeatherMap"
        }
    
    def _calculate_aqi_from_pm(self, pm25: float) -> float:
        """Calculate AQI from PM2.5 concentration"""
        if pm25 <= 12:
            return pm25 * 50 / 12
        elif pm25 <= 35.4:
            return 50 + (pm25 - 12) * 50 / 23.4
        elif pm25 <= 55.4:
            return 100 + (pm25 - 35.4) * 100 / 20
        elif pm25 <= 150.4:
            return 200 + (pm25 - 55.4) * 100 / 95
        elif pm25 <= 250.4:
            return 300 + (pm25 - 150.4) * 100 / 100
        else:
            return 400 + (pm25 - 250.4) * 100 / 250.4
    
    def _get_fallback_weather_data(self, pincode: str) -> Dict[str, Any]:
        """Fallback weather data"""
        import random
        return {
            "temperature": random.randint(20, 35),
            "humidity": random.randint(40, 80),
            "wind_speed": random.randint(5, 25),
            "pressure": random.randint(990, 1020),
            "pm25": random.randint(30, 120),
            "pm10": random.randint(50, 180),
            "aqi": random.randint(80, 250),
            "source": "fallback"
        }
    
    def _get_fallback_geocoding_data(self, pincode: str) -> Dict[str, Any]:
        """Fallback geocoding data"""
        city_coords = {
            "400001": {"lat": 19.0760, "lng": 72.8777, "city": "Mumbai"},
            "110001": {"lat": 28.6139, "lng": 77.2090, "city": "Delhi"},
            "560001": {"lat": 12.9716, "lng": 77.5946, "city": "Bangalore"},
            "411001": {"lat": 18.5204, "lng": 73.8567, "city": "Pune"},
        }
        
        coords = city_coords.get(pincode, {"lat": 20.0, "lng": 77.0, "city": "Unknown"})
        return {
            "latitude": coords["lat"],
            "longitude": coords["lng"],
            "city": coords["city"],
            "state": "India"
        }
    
    def _get_static_health_advisory(self, aqi_value: float) -> Dict[str, Any]:
        """Static health advisory based on AQI"""
        if aqi_value <= 50:
            return {
                "category": "Good",
                "message": "Air quality is excellent. Enjoy outdoor activities!",
                "precautions": ["No special precautions needed"],
                "risk_groups": ["None"],
                "health_effects": "No health concerns for general population"
            }
        elif aqi_value <= 100:
            return {
                "category": "Satisfactory", 
                "message": "Air quality is acceptable for most people.",
                "precautions": ["Sensitive individuals should limit prolonged outdoor exertion"],
                "risk_groups": ["People with respiratory conditions"],
                "health_effects": "Minor breathing discomfort for sensitive people"
            }
        elif aqi_value <= 200:
            return {
                "category": "Moderate",
                "message": "Air quality is unhealthy for sensitive groups.",
                "precautions": [
                    "Wear mask when going outdoors",
                    "Limit outdoor activities for children and elderly",
                    "Keep windows closed during peak hours"
                ],
                "risk_groups": ["Children", "Elderly", "People with heart/lung conditions"],
                "health_effects": "Breathing discomfort, coughing for sensitive groups"
            }
        elif aqi_value <= 300:
            return {
                "category": "Poor",
                "message": "Everyone may experience health effects.",
                "precautions": [
                    "Avoid outdoor activities",
                    "Use N95 masks outdoors", 
                    "Use air purifiers indoors",
                    "Keep windows and doors closed"
                ],
                "risk_groups": ["Everyone, especially children and elderly"],
                "health_effects": "Coughing, throat irritation, breathing difficulty"
            }
        elif aqi_value <= 400:
            return {
                "category": "Very Poor",
                "message": "Health alert: everyone may experience serious health effects.",
                "precautions": [
                    "Avoid all outdoor activities",
                    "Stay indoors with air purification",
                    "Wear N95/N99 masks if must go out",
                    "Consult doctor if experiencing symptoms"
                ],
                "risk_groups": ["Entire population"],
                "health_effects": "Serious aggravation of heart and lung conditions"
            }
        else:
            return {
                "category": "Severe",
                "message": "Emergency conditions: health warnings for everyone.",
                "precautions": [
                    "Stay indoors at all times",
                    "Emergency: avoid any outdoor exposure",
                    "Seek immediate medical attention if symptoms worsen",
                    "Use high-efficiency air purifiers"
                ],
                "risk_groups": ["Entire population at risk"],
                "health_effects": "Serious respiratory and cardiovascular effects"
            }
    
    def _parse_ai_response(self, response_text: str, aqi_value: float) -> Dict[str, Any]:
        """Parse AI response, fallback to static if parsing fails"""
        try:
            # Try to extract JSON from AI response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"AI response parsing error: {e}")
        
        # Fallback to static advisory
        return self._get_static_health_advisory(aqi_value)

# Global instance
external_api_service = ExternalAPIService()
