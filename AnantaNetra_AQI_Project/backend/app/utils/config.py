from pydantic_settings import BaseSettings
import os
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    weather_api_key: str
    openweather_api_key: str
    opencage_api_key: str
    gemini_api_key: str
    
    # Rate Limiting
    gemini_rpm_limit: int = 10
    gemini_rpd_limit: int = 250
    weather_delay_seconds: float = 3.6
    
    # Demo Mode
    demo_mode: bool = False
    demo_fixtures_path: str = "./tests/fixtures/demo_fixture.json"
    
    # Database
    database_url: str = "sqlite:///./aqi_monitoring.db"
    
    # Redis Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    
    # ML Models
    model_path: str = "./ml_models/"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate required API keys
        required_keys = [
            "weather_api_key", "openweather_api_key", 
            "opencage_api_key", "gemini_api_key"
        ]
        for key in required_keys:
            if not getattr(self, key):
                print(f"Warning: Missing environment variable: {key.upper()}")
                # Set demo mode if keys are missing
                self.demo_mode = True

settings = Settings()
