from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AQIResponse(BaseModel):
    aqi: float
    category: str
    pincode: str
    timestamp: str
    source: str
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None

class AQIRequest(BaseModel):
    pincode: str
    include_forecast: bool = False
