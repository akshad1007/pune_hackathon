from pydantic import BaseModel
from typing import List

class HealthAdvisoryResponse(BaseModel):
    category: str
    message: str
    precautions: List[str]
    risk_groups: List[str]
    aqi_range: str
    health_effects: str
