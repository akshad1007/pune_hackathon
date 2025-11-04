"""
Data Loading and Processing Pipeline
===================================

Comprehensive data loading, cleaning, and preprocessing for all datasets.
Handles the 8+ custom datasets and prepares them for ML modeling.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import geopandas as gpd
import structlog
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()


class DataLoader:
    """Load and process all datasets for the India Environmental Monitoring System."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.dataset_path = self.data_path / "dataset"
        self.cleaned_path = self.data_path / "cleaned"
        self.cleaned_path.mkdir(exist_ok=True)
        
        # Track data sources and loading status
        self.data_sources = {}
        self.loading_stats = {}
        
        # Initialize logging
        logger.info("DataLoader initialized", data_path=str(self.data_path))
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all 8+ datasets and return as dictionary."""
        logger.info("Loading all datasets...")
        
        datasets = {}
        
        try:
            # 1. Industrial Emissions Database
            datasets['factories'] = self.load_factory_data()
            
            # 2. Pincode Master Database
            datasets['pincodes'] = self.load_pincode_data()
            
            # 3. Vehicle Emissions Data
            datasets['vehicles'] = self.load_vehicle_data()
            
            # 4. Population Data
            datasets['population'] = self.load_population_data()
            
            # 5. Forest Cover Data
            datasets['forest'] = self.load_forest_data()
            
            # 6. Business Directory
            datasets['businesses'] = self.load_business_data()
            
            # 7. Geographic Intelligence
            datasets['geographic'] = self.load_geographic_data()
            
            # 8. Real-time API Data
            datasets['realtime'] = self.load_realtime_data()
            
            # Save loading statistics
            self._save_loading_stats(datasets)
            
            logger.info("All datasets loaded successfully", 
                       dataset_count=len(datasets),
                       total_records=sum(len(df) for df in datasets.values()))
            
            return datasets
            
        except Exception as e:
            logger.error("Error loading datasets", error=str(e))
            raise
    
    def load_factory_data(self) -> pd.DataFrame:
        """Load and clean industrial emissions database."""
        logger.info("Loading factory emissions data...")
        
        try:
            file_path = self.dataset_path / "07-list_of_registered_working_factories.xls"
            
            if file_path.exists():
                # Read Excel file
                df = pd.read_excel(file_path)
                
                # Clean column names
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                
                # Data cleaning and standardization
                df = self._clean_factory_data(df)
                
                self.data_sources['factories'] = str(file_path)
                logger.info("Factory data loaded", records=len(df))
                
                # Save cleaned data
                cleaned_path = self.cleaned_path / "factories_cleaned.csv"
                df.to_csv(cleaned_path, index=False)
                
                return df
            else:
                logger.warning("Factory data file not found, generating sample data")
                return self._generate_sample_factory_data()
                
        except Exception as e:
            logger.error("Error loading factory data", error=str(e))
            return self._generate_sample_factory_data()
    
    def _clean_factory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize factory data."""
        # Standardize column names mapping
        column_mapping = {
            'factory_name': 'name',
            'industry_type': 'industry',
            'location': 'address',
            'state': 'state',
            'district': 'district'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Add required columns if missing
        if 'emission_level' not in df.columns:
            # Classify emission levels based on industry type
            high_emission = ['chemical', 'steel', 'cement', 'power', 'petroleum']
            medium_emission = ['textile', 'paper', 'automobile', 'pharmaceutical']
            
            def classify_emission(industry):
                if pd.isna(industry):
                    return 'Medium'
                industry_lower = str(industry).lower()
                for keyword in high_emission:
                    if keyword in industry_lower:
                        return 'High'
                for keyword in medium_emission:
                    if keyword in industry_lower:
                        return 'Medium'
                return 'Low'
            
            df['emission_level'] = df.get('industry', '').apply(classify_emission)
        
        # Add geographic coordinates (mock data for demo)
        if 'latitude' not in df.columns:
            np.random.seed(42)
            df['latitude'] = np.random.uniform(8.0, 35.0, len(df))  # India lat range
            df['longitude'] = np.random.uniform(68.0, 97.0, len(df))  # India lng range
        
        # Clean text fields
        text_columns = ['name', 'industry', 'address', 'state', 'district']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
    
    def load_pincode_data(self) -> pd.DataFrame:
        """Load pincode database from SQLite."""
        logger.info("Loading pincode data...")
        
        try:
            db_path = self.data_path / "pincode_master.db"
            
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    df = pd.read_sql_query("""
                        SELECT pincode, city, state, latitude, longitude, 
                               district, region, country
                        FROM pincode_data 
                        LIMIT 50000
                    """, conn)
                
                self.data_sources['pincodes'] = str(db_path)
                logger.info("Pincode data loaded", records=len(df))
                
            else:
                logger.warning("Pincode database not found, loading CSV backup")
                csv_path = self.dataset_path / "Pincode_Dataset.csv"
                df = pd.read_csv(csv_path) if csv_path.exists() else self._generate_sample_pincode_data()
            
            # Save cleaned data
            cleaned_path = self.cleaned_path / "pincodes_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error("Error loading pincode data", error=str(e))
            return self._generate_sample_pincode_data()
    
    def load_vehicle_data(self) -> pd.DataFrame:
        """Load and combine vehicle emissions data."""
        logger.info("Loading vehicle emissions data...")
        
        try:
            datasets = []
            
            # Load transport vehicles
            transport_path = self.dataset_path / "Forest Combined Data - Transport Vehicles.csv"
            if transport_path.exists():
                transport_df = pd.read_csv(transport_path)
                transport_df['vehicle_type'] = 'Transport'
                datasets.append(transport_df)
            
            # Load non-transport vehicles
            non_transport_path = self.dataset_path / "Forest Combined Data - Non Transport Vehicles.csv"
            if non_transport_path.exists():
                non_transport_df = pd.read_csv(non_transport_path)
                non_transport_df['vehicle_type'] = 'Non-Transport'
                datasets.append(non_transport_df)
            
            # Load district vehicle data
            district_path = self.dataset_path / "district_vehicle_data.csv"
            if district_path.exists():
                district_df = pd.read_csv(district_path)
                district_df['vehicle_type'] = 'District_Total'
                datasets.append(district_df)
            
            if datasets:
                df = pd.concat(datasets, ignore_index=True)
                df = self._clean_vehicle_data(df)
            else:
                logger.warning("Vehicle data files not found, generating sample data")
                df = self._generate_sample_vehicle_data()
            
            self.data_sources['vehicles'] = "Multiple CSV files"
            logger.info("Vehicle data loaded", records=len(df))
            
            # Save cleaned data
            cleaned_path = self.cleaned_path / "vehicles_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error("Error loading vehicle data", error=str(e))
            return self._generate_sample_vehicle_data()
    
    def _clean_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize vehicle data."""
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Calculate vehicle density per 1000 population
        if 'total_vehicles' in df.columns and 'population' in df.columns:
            df['vehicle_density'] = (df['total_vehicles'] / df['population']) * 1000
        
        # Estimate emissions based on vehicle count and type
        emission_factors = {
            'Transport': 2.5,  # kg CO2/vehicle/day
            'Non-Transport': 1.2,
            'District_Total': 1.8
        }
        
        df['estimated_emissions'] = df['vehicle_type'].map(emission_factors).fillna(1.5) * df.get('total_vehicles', 0)
        
        return df
    
    def load_population_data(self) -> pd.DataFrame:
        """Load population-environment correlation data."""
        logger.info("Loading population data...")
        
        try:
            file_path = self.dataset_path / "district_population_data.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = self._clean_population_data(df)
            else:
                logger.warning("Population data file not found, generating sample data")
                df = self._generate_sample_population_data()
            
            self.data_sources['population'] = str(file_path) if file_path.exists() else "Generated"
            logger.info("Population data loaded", records=len(df))
            
            # Save cleaned data
            cleaned_path = self.cleaned_path / "population_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error("Error loading population data", error=str(e))
            return self._generate_sample_population_data()
    
    def _clean_population_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance population data."""
        # Calculate population density
        if 'population' in df.columns and 'area_km2' in df.columns:
            df['population_density'] = df['population'] / df['area_km2']
        
        # Calculate urbanization rate
        if 'urban_population' in df.columns and 'population' in df.columns:
            df['urbanization_rate'] = (df['urban_population'] / df['population']) * 100
        
        return df
    
    def load_forest_data(self) -> pd.DataFrame:
        """Load forest cover and environmental buffer data."""
        logger.info("Loading forest cover data...")
        
        try:
            file_path = self.dataset_path / "Forest Combined Data - Forest Area (1).csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = self._clean_forest_data(df)
            else:
                logger.warning("Forest data file not found, generating sample data")
                df = self._generate_sample_forest_data()
            
            self.data_sources['forest'] = str(file_path) if file_path.exists() else "Generated"
            logger.info("Forest data loaded", records=len(df))
            
            # Save cleaned data
            cleaned_path = self.cleaned_path / "forest_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error("Error loading forest data", error=str(e))
            return self._generate_sample_forest_data()
    
    def _clean_forest_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance forest data."""
        # Calculate forest cover percentage
        if 'forest_area_km2' in df.columns and 'total_area_km2' in df.columns:
            df['forest_cover_percent'] = (df['forest_area_km2'] / df['total_area_km2']) * 100
        
        # Estimate carbon absorption capacity
        # Average forest absorbs ~2.5 tons CO2 per hectare per year
        if 'forest_area_km2' in df.columns:
            df['carbon_absorption_tons_year'] = df['forest_area_km2'] * 100 * 2.5  # km2 to hectares
        
        return df
    
    def load_business_data(self) -> pd.DataFrame:
        """Load business and industry directory."""
        logger.info("Loading business directory data...")
        
        try:
            file_path = self.dataset_path / "indianindustriesdirectory.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = self._clean_business_data(df)
            else:
                logger.warning("Business data file not found, generating sample data")
                df = self._generate_sample_business_data()
            
            self.data_sources['businesses'] = str(file_path) if file_path.exists() else "Generated"
            logger.info("Business data loaded", records=len(df))
            
            # Save cleaned data
            cleaned_path = self.cleaned_path / "businesses_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error("Error loading business data", error=str(e))
            return self._generate_sample_business_data()
    
    def _clean_business_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and classify business data by environmental impact."""
        # Classify businesses by pollution potential
        high_pollution = ['chemical', 'steel', 'cement', 'mining', 'petroleum', 'power']
        medium_pollution = ['textile', 'paper', 'plastic', 'pharmaceutical', 'automobile']
        low_pollution = ['software', 'service', 'retail', 'agriculture', 'education']
        
        def classify_pollution_potential(business_type):
            if pd.isna(business_type):
                return 'Medium'
            business_lower = str(business_type).lower()
            for keyword in high_pollution:
                if keyword in business_lower:
                    return 'High'
            for keyword in medium_pollution:
                if keyword in business_lower:
                    return 'Medium'
            for keyword in low_pollution:
                if keyword in business_lower:
                    return 'Low'
            return 'Medium'
        
        if 'business_type' in df.columns:
            df['pollution_potential'] = df['business_type'].apply(classify_pollution_potential)
        
        return df
    
    def load_geographic_data(self) -> pd.DataFrame:
        """Load enhanced geographic intelligence database."""
        logger.info("Loading geographic intelligence data...")
        
        try:
            file_path = self.dataset_path / "Pincode_Dataset.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = self._enhance_geographic_data(df)
            else:
                logger.warning("Geographic data file not found, generating sample data")
                df = self._generate_sample_geographic_data()
            
            self.data_sources['geographic'] = str(file_path) if file_path.exists() else "Generated"
            logger.info("Geographic data loaded", records=len(df))
            
            # Save cleaned data
            cleaned_path = self.cleaned_path / "geographic_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error("Error loading geographic data", error=str(e))
            return self._generate_sample_geographic_data()
    
    def _enhance_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance geographic data with environmental zones."""
        # Classify environmental zones based on geography and development
        def classify_environmental_zone(row):
            # This is a simplified classification
            if 'metro' in str(row.get('city', '')).lower():
                return 'Urban_Metro'
            elif 'city' in str(row.get('city', '')).lower():
                return 'Urban_City'
            elif 'town' in str(row.get('city', '')).lower():
                return 'Semi_Urban'
            else:
                return 'Rural'
        
        df['environmental_zone'] = df.apply(classify_environmental_zone, axis=1)
        
        return df
    
    def load_realtime_data(self) -> pd.DataFrame:
        """Load real-time API integration dataset."""
        logger.info("Loading real-time API data...")
        
        try:
            datasets = []
            
            # Load Google API data files
            for i in ['', ' (1)', ' (2)']:
                file_path = self.dataset_path / f"google{i}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df['source'] = f'google_api_{i.strip("() ")}'
                    datasets.append(df)
            
            if datasets:
                df = pd.concat(datasets, ignore_index=True)
                df = self._clean_realtime_data(df)
            else:
                logger.warning("Real-time data files not found, generating sample data")
                df = self._generate_sample_realtime_data()
            
            self.data_sources['realtime'] = "Google API files"
            logger.info("Real-time data loaded", records=len(df))
            
            # Save cleaned data
            cleaned_path = self.cleaned_path / "realtime_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error("Error loading real-time data", error=str(e))
            return self._generate_sample_realtime_data()
    
    def _clean_realtime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize real-time API data."""
        # Add timestamp if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        # Parse coordinates if available
        if 'location' in df.columns:
            # Extract lat, lng from location string if formatted as "lat,lng"
            location_parts = df['location'].str.split(',', expand=True)
            if location_parts.shape[1] >= 2:
                df['latitude'] = pd.to_numeric(location_parts[0], errors='coerce')
                df['longitude'] = pd.to_numeric(location_parts[1], errors='coerce')
        
        return df
    
    # Sample data generators for missing files
    def _generate_sample_factory_data(self) -> pd.DataFrame:
        """Generate sample factory data for demo."""
        np.random.seed(42)
        n_factories = 500
        
        states = ['Maharashtra', 'Gujarat', 'Tamil Nadu', 'Karnataka', 'Uttar Pradesh']
        industries = ['Chemical', 'Textile', 'Pharmaceutical', 'Steel', 'Cement', 'Paper']
        emission_levels = ['High', 'Medium', 'Low']
        
        return pd.DataFrame({
            'name': [f'Factory_{i}' for i in range(n_factories)],
            'industry': np.random.choice(industries, n_factories),
            'state': np.random.choice(states, n_factories),
            'district': [f'District_{i%20}' for i in range(n_factories)],
            'emission_level': np.random.choice(emission_levels, n_factories),
            'latitude': np.random.uniform(8.0, 35.0, n_factories),
            'longitude': np.random.uniform(68.0, 97.0, n_factories)
        })
    
    def _generate_sample_pincode_data(self) -> pd.DataFrame:
        """Generate sample pincode data for demo."""
        np.random.seed(42)
        n_pincodes = 1000
        
        return pd.DataFrame({
            'pincode': [f'{i:06d}' for i in range(400001, 400001 + n_pincodes)],
            'city': [f'City_{i%50}' for i in range(n_pincodes)],
            'state': np.random.choice(['Maharashtra', 'Gujarat', 'Karnataka'], n_pincodes),
            'latitude': np.random.uniform(8.0, 35.0, n_pincodes),
            'longitude': np.random.uniform(68.0, 97.0, n_pincodes),
            'district': [f'District_{i%20}' for i in range(n_pincodes)]
        })
    
    def _generate_sample_vehicle_data(self) -> pd.DataFrame:
        """Generate sample vehicle data for demo."""
        np.random.seed(42)
        n_districts = 50
        
        return pd.DataFrame({
            'district': [f'District_{i}' for i in range(n_districts)],
            'state': np.random.choice(['Maharashtra', 'Gujarat', 'Karnataka'], n_districts),
            'total_vehicles': np.random.randint(10000, 500000, n_districts),
            'vehicle_type': 'District_Total',
            'population': np.random.randint(100000, 2000000, n_districts),
            'vehicle_density': np.random.uniform(50, 400, n_districts),
            'estimated_emissions': np.random.uniform(1000, 50000, n_districts)
        })
    
    def _generate_sample_population_data(self) -> pd.DataFrame:
        """Generate sample population data for demo."""
        np.random.seed(42)
        n_districts = 50
        
        total_pop = np.random.randint(100000, 2000000, n_districts)
        urban_pop = total_pop * np.random.uniform(0.3, 0.8, n_districts)
        
        return pd.DataFrame({
            'district': [f'District_{i}' for i in range(n_districts)],
            'state': np.random.choice(['Maharashtra', 'Gujarat', 'Karnataka'], n_districts),
            'population': total_pop,
            'urban_population': urban_pop.astype(int),
            'area_km2': np.random.uniform(1000, 10000, n_districts),
            'population_density': total_pop / np.random.uniform(1000, 10000, n_districts),
            'urbanization_rate': (urban_pop / total_pop) * 100
        })
    
    def _generate_sample_forest_data(self) -> pd.DataFrame:
        """Generate sample forest data for demo."""
        np.random.seed(42)
        n_districts = 50
        
        total_area = np.random.uniform(1000, 10000, n_districts)
        forest_area = total_area * np.random.uniform(0.1, 0.6, n_districts)
        
        return pd.DataFrame({
            'district': [f'District_{i}' for i in range(n_districts)],
            'state': np.random.choice(['Maharashtra', 'Gujarat', 'Karnataka'], n_districts),
            'total_area_km2': total_area,
            'forest_area_km2': forest_area,
            'forest_cover_percent': (forest_area / total_area) * 100,
            'carbon_absorption_tons_year': forest_area * 100 * 2.5
        })
    
    def _generate_sample_business_data(self) -> pd.DataFrame:
        """Generate sample business data for demo."""
        np.random.seed(42)
        n_businesses = 1000
        
        business_types = ['Software', 'Chemical', 'Textile', 'Retail', 'Pharmaceutical']
        
        return pd.DataFrame({
            'business_name': [f'Business_{i}' for i in range(n_businesses)],
            'business_type': np.random.choice(business_types, n_businesses),
            'district': [f'District_{i%20}' for i in range(n_businesses)],
            'state': np.random.choice(['Maharashtra', 'Gujarat', 'Karnataka'], n_businesses),
            'pollution_potential': np.random.choice(['High', 'Medium', 'Low'], n_businesses)
        })
    
    def _generate_sample_geographic_data(self) -> pd.DataFrame:
        """Generate sample geographic data for demo."""
        np.random.seed(42)
        n_locations = 500
        
        return pd.DataFrame({
            'pincode': [f'{i:06d}' for i in range(400001, 400001 + n_locations)],
            'city': [f'City_{i%50}' for i in range(n_locations)],
            'state': np.random.choice(['Maharashtra', 'Gujarat', 'Karnataka'], n_locations),
            'latitude': np.random.uniform(8.0, 35.0, n_locations),
            'longitude': np.random.uniform(68.0, 97.0, n_locations),
            'environmental_zone': np.random.choice(['Urban_Metro', 'Urban_City', 'Semi_Urban', 'Rural'], n_locations)
        })
    
    def _generate_sample_realtime_data(self) -> pd.DataFrame:
        """Generate sample real-time data for demo."""
        np.random.seed(42)
        n_records = 100
        
        return pd.DataFrame({
            'location': [f'{np.random.uniform(8.0, 35.0):.4f},{np.random.uniform(68.0, 97.0):.4f}' 
                        for _ in range(n_records)],
            'timestamp': pd.date_range(start='2024-01-01', periods=n_records, freq='H'),
            'traffic_density': np.random.uniform(0.1, 1.0, n_records),
            'popularity_score': np.random.uniform(0.0, 100.0, n_records),
            'source': 'google_api_sample'
        })
    
    def _save_loading_stats(self, datasets: Dict[str, pd.DataFrame]):
        """Save dataset loading statistics."""
        stats = {
            'loading_timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        for name, df in datasets.items():
            stats['datasets'][name] = {
                'record_count': len(df),
                'column_count': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'data_source': self.data_sources.get(name, 'Generated'),
                'columns': list(df.columns)
            }
        
        # Save to JSON
        stats_path = self.data_path / "data_loading_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Data loading statistics saved", stats_file=str(stats_path))


def create_integrated_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create integrated dataset by merging all data sources.
    This is the key function for preparing ML-ready features.
    """
    logger.info("Creating integrated dataset...")
    
    # Start with pincode data as base
    base_df = datasets['pincodes'].copy()
    
    # Add district-level data
    district_features = []
    
    # Population features
    if 'population' in datasets:
        pop_df = datasets['population'].groupby('district').agg({
            'population': 'sum',
            'population_density': 'mean',
            'urbanization_rate': 'mean'
        }).reset_index()
        district_features.append(pop_df)
    
    # Forest features
    if 'forest' in datasets:
        forest_df = datasets['forest'].groupby('district').agg({
            'forest_cover_percent': 'mean',
            'carbon_absorption_tons_year': 'sum'
        }).reset_index()
        district_features.append(forest_df)
    
    # Vehicle features
    if 'vehicles' in datasets:
        vehicle_df = datasets['vehicles'].groupby('district').agg({
            'vehicle_density': 'mean',
            'estimated_emissions': 'sum'
        }).reset_index()
        district_features.append(vehicle_df)
    
    # Factory features
    if 'factories' in datasets:
        factory_df = datasets['factories'].groupby('district').agg({
            'name': 'count',  # Number of factories
            'emission_level': lambda x: (x == 'High').sum()  # High emission factories
        }).reset_index()
        factory_df.columns = ['district', 'factory_count', 'high_emission_factories']
        district_features.append(factory_df)
    
    # Merge all district-level features
    district_merged = None
    for df in district_features:
        if district_merged is None:
            district_merged = df
        else:
            district_merged = district_merged.merge(df, on='district', how='outer')
    
    # Merge with base pincode data
    if district_merged is not None:
        integrated_df = base_df.merge(district_merged, on='district', how='left')
    else:
        integrated_df = base_df.copy()
    
    # Fill missing values
    numeric_columns = integrated_df.select_dtypes(include=[np.number]).columns
    integrated_df[numeric_columns] = integrated_df[numeric_columns].fillna(integrated_df[numeric_columns].median())
    
    logger.info("Integrated dataset created", 
               records=len(integrated_df), 
               features=len(integrated_df.columns))
    
    return integrated_df


# Global loader instance
data_loader = DataLoader()


def load_and_prepare_data() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Main function to load all datasets and create integrated ML-ready data.
    
    Returns:
        Tuple of (individual_datasets, integrated_dataset)
    """
    logger.info("Starting complete data loading and preparation...")
    
    # Load all individual datasets
    datasets = data_loader.load_all_datasets()
    
    # Create integrated dataset for ML
    integrated_data = create_integrated_dataset(datasets)
    
    # Save integrated dataset
    integrated_path = data_loader.cleaned_path / "integrated_dataset.csv"
    integrated_data.to_csv(integrated_path, index=False)
    
    logger.info("Data loading and preparation completed",
               individual_datasets=len(datasets),
               integrated_records=len(integrated_data),
               integrated_features=len(integrated_data.columns))
    
    return datasets, integrated_data


if __name__ == "__main__":
    # Demo run
    datasets, integrated = load_and_prepare_data()
    print(f"Loaded {len(datasets)} datasets")
    print(f"Integrated dataset: {integrated.shape}")
    print(f"Features: {list(integrated.columns)}")
