"""
Feature Engineering Pipeline
===========================

Advanced feature engineering for India Environmental Monitoring System.
Creates lag features, rolling statistics, spatial features, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import structlog

logger = structlog.get_logger()


class FeatureEngineer:
    """Advanced feature engineering for environmental data."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.feature_importance = {}
        
        logger.info("Feature Engineer initialized")
    
    def create_all_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create comprehensive feature set from raw data.
        
        Args:
            df: Input dataframe with environmental data
            target_col: Target column name for supervised learning
        
        Returns:
            Dataframe with engineered features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # Make a copy to avoid modifying original
        featured_df = df.copy()
        
        # 1. Temporal features
        if 'timestamp' in df.columns or any('date' in col.lower() for col in df.columns):
            featured_df = self.create_temporal_features(featured_df)
            logger.info("Temporal features created")
        
        # 2. Lag features (if we have time series data)
        if 'timestamp' in featured_df.columns and target_col and target_col in featured_df.columns:
            featured_df = self.create_lag_features(featured_df, target_col)
            logger.info("Lag features created")
        
        # 3. Rolling window features
        numeric_cols = featured_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            featured_df = self.create_rolling_features(featured_df, numeric_cols)
            logger.info("Rolling window features created")
        
        # 4. Interaction features
        featured_df = self.create_interaction_features(featured_df)
        logger.info("Interaction features created")
        
        # 5. Spatial features (if coordinates available)
        if 'latitude' in featured_df.columns and 'longitude' in featured_df.columns:
            featured_df = self.create_spatial_features(featured_df)
            logger.info("Spatial features created")
        
        # 6. Environmental zone features
        featured_df = self.create_environmental_features(featured_df)
        logger.info("Environmental features created")
        
        # 7. Categorical encoding
        featured_df = self.encode_categorical_features(featured_df)
        logger.info("Categorical features encoded")
        
        # 8. Pollution source proximity features
        featured_df = self.create_pollution_proximity_features(featured_df)
        logger.info("Pollution proximity features created")
        
        # 9. Weather-based derived features
        featured_df = self.create_weather_derived_features(featured_df)
        logger.info("Weather-derived features created")
        
        # 10. Statistical aggregation features
        featured_df = self.create_statistical_features(featured_df)
        logger.info("Statistical features created")
        
        # Store feature names
        self.feature_names = list(featured_df.columns)
        
        # Feature selection (if target is provided)
        if target_col and target_col in featured_df.columns:
            featured_df = self.select_important_features(featured_df, target_col)
            logger.info("Feature selection completed")
        
        logger.info("Feature engineering completed", 
                   original_features=len(df.columns),
                   engineered_features=len(featured_df.columns),
                   new_features=len(featured_df.columns) - len(df.columns))
        
        return featured_df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp columns."""
        # Find timestamp column
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col is None:
            # Create dummy timestamp for demo
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            timestamp_col = 'timestamp'
        
        # Ensure datetime type
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract temporal features
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
        df['quarter'] = df[timestamp_col].dt.quarter
        df['is_weekend'] = (df[timestamp_col].dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)
        
        # Cyclical encoding for better ML performance
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.hour / 24)
        
        # Seasonal indicators
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                                       9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        
        # Indian seasonal patterns
        df['monsoon_season'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        df['winter_pollution_season'] = df['month'].isin([11, 12, 1, 2]).astype(int)
        df['crop_burning_season'] = df['month'].isin([10, 11]).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           lags: List[int] = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for time series prediction."""
        if 'timestamp' not in df.columns:
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create lag features for target variable
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Create lag features for key predictors
        key_predictors = ['temperature', 'humidity', 'wind_speed', 'pressure']
        for predictor in key_predictors:
            if predictor in df.columns:
                for lag in [1, 7]:  # Just short lags for predictors
                    df[f'{predictor}_lag_{lag}'] = df[predictor].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, numeric_cols: List[str],
                               windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """Create rolling window statistics."""
        if 'timestamp' not in df.columns:
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Key columns for rolling features
        key_cols = ['temperature', 'humidity', 'wind_speed', 'pressure']
        if hasattr(self, 'target_col') and self.target_col:
            key_cols.append(self.target_col)
        
        rolling_cols = [col for col in key_cols if col in numeric_cols]
        
        for col in rolling_cols[:5]:  # Limit to avoid too many features
            for window in windows:
                if len(df) > window:  # Only if we have enough data
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling standard deviation
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    
                    # Rolling min/max
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        
        # Weather interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['heat_index'] = self.calculate_heat_index(df.get('temperature', 25), df.get('humidity', 60))
        
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_chill_factor'] = df['wind_speed'] * (20 - df['temperature'])
        
        # Pollution source interactions
        if 'vehicle_density' in df.columns and 'population_density' in df.columns:
            df['vehicle_population_ratio'] = df['vehicle_density'] / (df['population_density'] + 1)
        
        if 'factory_count' in df.columns and 'wind_speed' in df.columns:
            df['industrial_dispersion_factor'] = df['factory_count'] / (df['wind_speed'] + 1)
        
        # Environmental buffer interactions
        if 'forest_cover_percent' in df.columns and 'vehicle_density' in df.columns:
            df['green_cover_vehicle_ratio'] = df['forest_cover_percent'] / (df['vehicle_density'] + 1)
        
        # Urban development interactions
        if 'urbanization_rate' in df.columns and 'population_density' in df.columns:
            df['urban_density_index'] = df['urbanization_rate'] * df['population_density'] / 100
        
        return df
    
    def calculate_heat_index(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity."""
        # Simplified heat index calculation
        T = temperature
        H = humidity
        
        # Heat index formula (simplified)
        HI = (T + H) / 2 + (T - 32) * 0.1
        
        return HI
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features from latitude/longitude."""
        
        # Distance to major cities (approximate coordinates)
        major_cities = {
            'delhi': (28.7041, 77.1025),
            'mumbai': (19.0760, 72.8777),
            'bangalore': (12.9716, 77.5946),
            'chennai': (13.0827, 80.2707),
            'kolkata': (22.5726, 88.3639),
            'hyderabad': (17.3850, 78.4867)
        }
        
        for city, (city_lat, city_lon) in major_cities.items():
            df[f'distance_to_{city}'] = self.haversine_distance(
                df['latitude'], df['longitude'], city_lat, city_lon
            )
        
        # Spatial clustering features
        df['lat_cluster'] = pd.cut(df['latitude'], bins=10, labels=False)
        df['lon_cluster'] = pd.cut(df['longitude'], bins=10, labels=False)
        df['spatial_cluster'] = df['lat_cluster'] * 10 + df['lon_cluster']
        
        # Coastal proximity (simplified)
        coastal_lat_ranges = [(8, 12), (12, 16), (16, 22)]  # Rough coastal regions
        df['coastal_proximity'] = 0
        for lat_min, lat_max in coastal_lat_ranges:
            coastal_mask = (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)
            df.loc[coastal_mask, 'coastal_proximity'] = 1
        
        # Elevation proxy (latitude-based approximation)
        df['elevation_proxy'] = np.maximum(0, (df['latitude'] - 15) * 100)
        
        return df
    
    def haversine_distance(self, lat1: pd.Series, lon1: pd.Series, 
                          lat2: float, lon2: float) -> pd.Series:
        """Calculate haversine distance between points."""
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        R = 6371
        distance = R * c
        
        return distance
    
    def create_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to environmental conditions."""
        
        # Air quality risk factors
        risk_factors = []
        
        if 'vehicle_density' in df.columns:
            df['high_vehicle_density'] = (df['vehicle_density'] > df['vehicle_density'].quantile(0.75)).astype(int)
            risk_factors.append('high_vehicle_density')
        
        if 'factory_count' in df.columns:
            df['high_industrial_activity'] = (df['factory_count'] > df['factory_count'].quantile(0.75)).astype(int)
            risk_factors.append('high_industrial_activity')
        
        if 'population_density' in df.columns:
            df['high_population_density'] = (df['population_density'] > df['population_density'].quantile(0.75)).astype(int)
            risk_factors.append('high_population_density')
        
        if 'forest_cover_percent' in df.columns:
            df['low_forest_cover'] = (df['forest_cover_percent'] < df['forest_cover_percent'].quantile(0.25)).astype(int)
            risk_factors.append('low_forest_cover')
        
        # Combined risk score
        if risk_factors:
            df['environmental_risk_score'] = df[risk_factors].sum(axis=1)
        
        # Environmental quality index
        quality_factors = []
        
        if 'forest_cover_percent' in df.columns:
            df['forest_cover_normalized'] = df['forest_cover_percent'] / 100
            quality_factors.append('forest_cover_normalized')
        
        if 'wind_speed' in df.columns:
            df['wind_dispersion_factor'] = np.log1p(df['wind_speed'])  # Log transform for wind
            quality_factors.append('wind_dispersion_factor')
        
        if quality_factors:
            df['environmental_quality_index'] = df[quality_factors].mean(axis=1)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for ML models."""
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # One-hot encode low cardinality
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=f'{col}', drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                
                # Store encoder info
                self.encoders[col] = {'type': 'onehot', 'categories': df[col].unique()}
                
            else:  # Label encode high cardinality
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                
                # Store encoder
                self.encoders[col] = {'type': 'label', 'encoder': le}
        
        # Drop original categorical columns
        df = df.drop(columns=categorical_cols)
        
        return df
    
    def create_pollution_proximity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on proximity to pollution sources."""
        
        # Factory proximity features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # For demo, create synthetic factory locations
            if 'factory_distance' not in df.columns:
                # Random factory locations for demo
                np.random.seed(42)
                n_factories = 50
                factory_lats = np.random.uniform(df['latitude'].min(), df['latitude'].max(), n_factories)
                factory_lons = np.random.uniform(df['longitude'].min(), df['longitude'].max(), n_factories)
                
                # Calculate minimum distance to any factory
                min_distances = []
                for idx, row in df.iterrows():
                    distances = self.haversine_distance(
                        pd.Series([row['latitude']] * n_factories),
                        pd.Series([row['longitude']] * n_factories),
                        factory_lats[0], factory_lons[0]
                    )
                    for i in range(1, n_factories):
                        dist = self.haversine_distance(
                            pd.Series([row['latitude']]),
                            pd.Series([row['longitude']]),
                            factory_lats[i], factory_lons[i]
                        )[0]
                        distances = pd.concat([distances, pd.Series([dist])])
                    
                    min_distances.append(distances.min())
                
                df['min_factory_distance'] = min_distances
        
        # Highway proximity (simplified)
        if 'latitude' in df.columns:
            # Major highway approximation (simplified north-south corridors)
            major_highways_lon = [77.1, 72.8, 80.2, 78.5]  # Delhi, Mumbai, Chennai, Hyderabad longitudes
            
            min_highway_distances = []
            for idx, row in df.iterrows():
                distances = [abs(row['longitude'] - hw_lon) for hw_lon in major_highways_lon]
                min_highway_distances.append(min(distances) * 111)  # Convert to km approximately
            
            df['min_highway_distance'] = min_highway_distances
        
        # Urban center proximity
        if 'population_density' in df.columns:
            df['urban_center_proximity'] = np.log1p(df['population_density'])
        
        return df
    
    def create_weather_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from weather data."""
        
        # Atmospheric stability indicators
        if 'temperature' in df.columns and 'pressure' in df.columns:
            df['atmospheric_stability'] = df['pressure'] / (df['temperature'] + 273.15)  # Simplified
        
        # Pollution dispersion potential
        if 'wind_speed' in df.columns and 'humidity' in df.columns:
            df['dispersion_potential'] = df['wind_speed'] / (df['humidity'] / 100 + 0.1)
        
        # Weather-based pollution risk
        weather_risk_factors = []
        
        if 'wind_speed' in df.columns:
            df['low_wind_risk'] = (df['wind_speed'] < df['wind_speed'].quantile(0.25)).astype(int)
            weather_risk_factors.append('low_wind_risk')
        
        if 'humidity' in df.columns:
            df['high_humidity_risk'] = (df['humidity'] > df['humidity'].quantile(0.75)).astype(int)
            weather_risk_factors.append('high_humidity_risk')
        
        if 'temperature' in df.columns:
            df['extreme_temp_risk'] = ((df['temperature'] < df['temperature'].quantile(0.1)) | 
                                      (df['temperature'] > df['temperature'].quantile(0.9))).astype(int)
            weather_risk_factors.append('extreme_temp_risk')
        
        if weather_risk_factors:
            df['weather_pollution_risk'] = df[weather_risk_factors].sum(axis=1)
        
        # Visibility and clarity indicators
        if 'humidity' in df.columns and 'temperature' in df.columns:
            df['visibility_index'] = (100 - df['humidity']) * np.log1p(df['temperature'])
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features."""
        
        # Group-based statistics
        if 'district' in df.columns:
            district_stats = df.groupby('district').agg({
                col: ['mean', 'std', 'min', 'max'] 
                for col in df.select_dtypes(include=[np.number]).columns[:5]
            }).reset_index()
            
            # Flatten column names
            district_stats.columns = ['district'] + [f'{col[0]}_district_{col[1]}' for col in district_stats.columns[1:]]
            
            # Merge back
            df = df.merge(district_stats, on='district', how='left', suffixes=('', '_district_stat'))
        
        # Percentile rankings
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        key_cols = ['temperature', 'humidity', 'wind_speed', 'vehicle_density', 'population_density']
        
        for col in key_cols:
            if col in numeric_cols:
                df[f'{col}_percentile'] = df[col].rank(pct=True)
        
        # Z-scores (standardized values)
        for col in key_cols[:3]:  # Limit to avoid too many features
            if col in numeric_cols:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        return df
    
    def select_important_features(self, df: pd.DataFrame, target_col: str, 
                                 k: int = 50) -> pd.DataFrame:
        """Select most important features using statistical tests."""
        
        if target_col not in df.columns:
            return df
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Select numeric features only for feature selection
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]
        
        if len(X_numeric.columns) <= k:
            return df  # Return original if we have fewer features than k
        
        try:
            # Use mutual information for feature selection
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(X_numeric.columns)))
            X_selected = selector.fit_transform(X_numeric, y)
            
            # Get selected feature names
            selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            # Store feature importance scores
            feature_scores = selector.scores_
            self.feature_importance = dict(zip(X_numeric.columns, feature_scores))
            
            # Return dataframe with selected features plus target
            selected_df = df[selected_features + [target_col]]
            
            logger.info("Feature selection completed",
                       original_features=len(X_numeric.columns),
                       selected_features=len(selected_features))
            
            return selected_df
            
        except Exception as e:
            logger.warning("Feature selection failed, returning original data", error=str(e))
            return df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from last feature selection."""
        return self.feature_importance
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        return self.feature_names


def create_ml_ready_features(df: pd.DataFrame, target_col: str = 'aqi') -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Main function to create ML-ready features from raw environmental data.
    
    Args:
        df: Raw environmental data
        target_col: Target variable column name
    
    Returns:
        Tuple of (featured_dataframe, feature_engineer_instance)
    """
    logger.info("Creating ML-ready features...")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create comprehensive features
    featured_df = engineer.create_all_features(df, target_col)
    
    # Remove rows with too many missing values
    missing_threshold = 0.5  # Remove rows missing >50% of values
    featured_df = featured_df.dropna(thresh=len(featured_df.columns) * (1 - missing_threshold))
    
    # Fill remaining missing values
    numeric_cols = featured_df.select_dtypes(include=[np.number]).columns
    featured_df[numeric_cols] = featured_df[numeric_cols].fillna(featured_df[numeric_cols].median())
    
    # Remove constant features
    constant_features = [col for col in featured_df.columns 
                        if featured_df[col].nunique() <= 1]
    if constant_features:
        featured_df = featured_df.drop(columns=constant_features)
        logger.info("Removed constant features", count=len(constant_features))
    
    logger.info("ML-ready features created",
               final_features=len(featured_df.columns),
               final_samples=len(featured_df))
    
    return featured_df, engineer


if __name__ == "__main__":
    # Demo run
    from data_processing import load_and_prepare_data
    
    datasets, integrated = load_and_prepare_data()
    
    # Add synthetic AQI target for demo
    if 'aqi' not in integrated.columns:
        np.random.seed(42)
        integrated['aqi'] = np.random.normal(120, 40, len(integrated))
        integrated['aqi'] = np.clip(integrated['aqi'], 0, 500)
    
    featured_df, engineer = create_ml_ready_features(integrated, 'aqi')
    
    print(f"Feature engineering completed!")
    print(f"Original features: {len(integrated.columns)}")
    print(f"Engineered features: {len(featured_df.columns)}")
    print(f"Feature importance scores: {len(engineer.get_feature_importance())}")
    
    # Show top features
    importance = engineer.get_feature_importance()
    if importance:
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most important features:")
        for i, (feature, score) in enumerate(top_features, 1):
            print(f"{i}. {feature}: {score:.4f}")
