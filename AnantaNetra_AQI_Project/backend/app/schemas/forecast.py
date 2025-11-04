from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class ForecastResponse(BaseModel):
    timestamp: str
    predicted_aqi: float
    category: str
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    pincode: str

class ForecastRequest(BaseModel):
    pincode: str
    hours_ahead: int = 24
    include_confidence: bool = True
