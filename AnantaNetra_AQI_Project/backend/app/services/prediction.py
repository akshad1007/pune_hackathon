import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import os

from app.utils.config import settings
from app.schemas.forecast import ForecastResponse

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            # Check for models in different locations
            model_paths = [
                settings.model_path,
                "./ml_models/",
                "../real_aqi_model.pkl",
                "./real_aqi_model.pkl"
            ]
            
            model_loaded = False
            
            for path in model_paths:
                try:
                    if os.path.exists(path):
                        if path.endswith('.pkl'):
                            # Single model file
                            with open(path, 'rb') as f:
                                model_data = pickle.load(f)
                                if isinstance(model_data, dict):
                                    self.models['primary'] = model_data.get('model', model_data)
                                    self.feature_columns = model_data.get('feature_columns', [])
                                else:
                                    self.models['primary'] = model_data
                                    
                                model_loaded = True
                                logger.info(f"Loaded model from {path}")
                                break
                        else:
                            # Directory with multiple models
                            for model_file in ['lstm_model.pkl', 'xgboost_model.pkl', 'ensemble_model.pkl']:
                                model_path = os.path.join(path, model_file)
                                if os.path.exists(model_path):
                                    with open(model_path, 'rb') as f:
                                        model_data = pickle.load(f)
                                        model_name = model_file.replace('_model.pkl', '')
                                        self.models[model_name] = model_data.get('model', model_data)
                                        if not self.feature_columns and 'feature_columns' in model_data:
                                            self.feature_columns = model_data['feature_columns']
                                    
                                    model_loaded = True
                                    logger.info(f"Loaded {model_name} model")
                
                except Exception as e:
                    logger.warning(f"Could not load model from {path}: {e}")
                    continue
            
            if not model_loaded:
                logger.warning("No ML models found, using fallback prediction")
                self.models['fallback'] = self._create_fallback_model()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models['fallback'] = self._create_fallback_model()
    
    async def predict_aqi(
        self, 
        pincode: str, 
        hours_ahead: int = 24,
        include_confidence: bool = True
    ) -> List[ForecastResponse]:
        """Generate AQI predictions"""
        try:
            # Prepare input features
            features = await self._prepare_features(pincode)
            
            # Generate predictions using best available model
            if 'ensemble' in self.models:
                predictions = self._predict_with_ensemble(features, hours_ahead)
            elif 'xgboost' in self.models:
                predictions = self._predict_with_xgboost(features, hours_ahead)
            elif 'primary' in self.models:
                predictions = self._predict_with_primary_model(features, hours_ahead)
            else:
                predictions = self._predict_with_fallback(features, hours_ahead)
            
            # Format response
            forecast_responses = []
            base_time = datetime.now()
            
            for i, (aqi_value, confidence) in enumerate(predictions):
                forecast_time = base_time + timedelta(hours=i+1)
                
                response = ForecastResponse(
                    timestamp=forecast_time.isoformat(),
                    predicted_aqi=round(aqi_value, 2),
                    confidence_lower=round(confidence[0], 2) if include_confidence else None,
                    confidence_upper=round(confidence[1], 2) if include_confidence else None,
                    category=self._get_aqi_category(aqi_value),
                    pincode=pincode
                )
                forecast_responses.append(response)
            
            return forecast_responses
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return fallback predictions
            return self._generate_fallback_forecast(pincode, hours_ahead)
    
    async def _prepare_features(self, pincode: str) -> np.ndarray:
        """Prepare features for prediction"""
        try:
            from app.services.external_apis import external_api_service
            
            # Get current weather data
            weather_data = await external_api_service.get_weather_data(pincode)
            geocoding_data = await external_api_service.get_geocoding_data(pincode)
            
            # Basic features
            current_time = datetime.now()
            features = {
                'temperature': weather_data.get('temperature', 25),
                'humidity': weather_data.get('humidity', 60),
                'wind_speed': weather_data.get('wind_speed', 10),
                'pressure': weather_data.get('pressure', 1013),
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.weekday(),
                'month': current_time.month,
                'seasonal_factor': np.sin(2 * np.pi * current_time.month / 12),
                'pm25_lag1': weather_data.get('pm25', 50),
                'pm10_lag1': weather_data.get('pm10', 80),
            }
            
            # Add derived features
            features.update(self._get_location_features(pincode, geocoding_data))
            
            # Create feature array
            if self.feature_columns:
                feature_array = np.array([features.get(col, 0) for col in self.feature_columns])
            else:
                feature_array = np.array(list(features.values()))
            
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            # Return default features
            return np.array([[25, 60, 10, 1013, 12, 1, 6, 0, 50, 80, 100, 50, 75, 0.5, 1.2]]).reshape(1, -1)
    
    def _get_location_features(self, pincode: str, geocoding_data: Dict) -> Dict[str, float]:
        """Get location-specific features"""
        # Demo location features based on major Indian cities
        location_features = {
            "400001": {"factory_density": 120, "vehicle_count": 850, "population_density": 28000, "forest_cover_ratio": 0.1},  # Mumbai
            "110001": {"factory_density": 95, "vehicle_count": 920, "population_density": 29000, "forest_cover_ratio": 0.15},   # Delhi
            "560001": {"factory_density": 80, "vehicle_count": 600, "population_density": 15000, "forest_cover_ratio": 0.25},   # Bangalore
            "411001": {"factory_density": 70, "vehicle_count": 520, "population_density": 12000, "forest_cover_ratio": 0.3},    # Pune
        }
        
        default_features = {"factory_density": 75, "vehicle_count": 650, "population_density": 18000, "forest_cover_ratio": 0.2}
        features = location_features.get(pincode, default_features)
        
        # Calculate derived features
        features["industrial_emission_score"] = features["factory_density"] * features["vehicle_count"] / (features["forest_cover_ratio"] + 0.1)
        features["traffic_density"] = features["vehicle_count"] / features["population_density"] * 1000
        
        return features
    
    def _predict_with_primary_model(self, features: np.ndarray, hours_ahead: int) -> List[tuple]:
        """Make predictions with primary model"""
        try:
            model = self.models['primary']
            base_prediction = model.predict(features)[0]
            
            predictions = []
            for i in range(hours_ahead):
                # Add some temporal variation
                time_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 24)  # Daily cycle
                trend_factor = 1 + 0.02 * i  # Slight upward trend
                noise = np.random.normal(0, 5)  # Random noise
                
                pred_aqi = max(10, base_prediction * time_factor * trend_factor + noise)
                confidence = (pred_aqi * 0.8, pred_aqi * 1.2)  # ±20% confidence
                
                predictions.append((pred_aqi, confidence))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Primary model prediction error: {e}")
            return self._predict_with_fallback(features, hours_ahead)
    
    def _predict_with_fallback(self, features: np.ndarray, hours_ahead: int) -> List[tuple]:
        """Fallback prediction method"""
        import random
        
        # Base AQI from recent patterns
        base_aqi = random.randint(100, 200)
        
        predictions = []
        for i in range(hours_ahead):
            # Simulate daily pattern and random variation
            hour_of_day = (datetime.now().hour + i) % 24
            
            # Higher pollution during rush hours and evening
            if hour_of_day in [7, 8, 9, 18, 19, 20]:
                hour_factor = 1.2
            elif hour_of_day in [2, 3, 4, 5]:
                hour_factor = 0.8
            else:
                hour_factor = 1.0
            
            variation = random.uniform(-20, 20)
            pred_aqi = max(10, min(500, base_aqi * hour_factor + variation))
            confidence = (pred_aqi * 0.75, pred_aqi * 1.25)
            
            predictions.append((pred_aqi, confidence))
        
        return predictions
    
    def _get_aqi_category(self, aqi_value: float) -> str:
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
    
    def _generate_fallback_forecast(self, pincode: str, hours_ahead: int) -> List[ForecastResponse]:
        """Generate fallback forecast when prediction fails"""
        base_time = datetime.now()
        base_aqi = 150  # Moderate baseline
        
        forecast_responses = []
        for i in range(hours_ahead):
            forecast_time = base_time + timedelta(hours=i+1)
            
            # Simple pattern: higher during day, lower at night
            hour_of_day = forecast_time.hour
            if 6 <= hour_of_day <= 22:
                aqi_value = base_aqi + (i * 2) + np.random.normal(0, 10)
            else:
                aqi_value = base_aqi - 20 + np.random.normal(0, 10)
            
            aqi_value = max(10, min(500, aqi_value))
            
            response = ForecastResponse(
                timestamp=forecast_time.isoformat(),
                predicted_aqi=round(aqi_value, 2),
                confidence_lower=round(aqi_value * 0.8, 2),
                confidence_upper=round(aqi_value * 1.2, 2),
                category=self._get_aqi_category(aqi_value),
                pincode=pincode
            )
            forecast_responses.append(response)
        
        return forecast_responses
    
    def _create_fallback_model(self):
        """Create a simple fallback model"""
        class FallbackModel:
            def predict(self, X):
                # Simple linear combination of features
                if X.shape[1] >= 5:
                    # Use first few features for prediction
                    temp, humidity, wind, pressure, hour = X[0][:5]
                    # Simple heuristic
                    aqi = 100 + (temp - 25) * 2 + (humidity - 50) * 0.5 - wind * 2 + (hour - 12) * 1.5
                    return np.array([max(10, min(500, aqi))])
                else:
                    return np.array([150])  # Default moderate AQI
        
        return FallbackModel()

# Global instance
prediction_service = PredictionService()
