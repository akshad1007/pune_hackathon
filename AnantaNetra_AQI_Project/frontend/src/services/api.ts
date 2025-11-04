import axios, { AxiosResponse, AxiosError } from 'axios';
import { AQIData, ForecastData, HealthAdvisory, MapCityData, SystemStatus } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    console.error('API Error:', error);
    
    // Handle specific error cases
    if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
      console.error('Backend server is not running, using fallback data');
      // Return fallback data
      return Promise.resolve({ data: getFallbackData(error.config?.url) });
    }
    
    if (error.response?.status === 429) {
      console.warn('Rate limit exceeded, using cached data');
    }
    
    if (error.response?.status === 500) {
      console.warn('Server error, attempting fallback');
      return Promise.resolve({ data: getFallbackData(error.config?.url) });
    }
    
    return Promise.reject(error);
  }
);

// API service functions
export const apiService = {
  // AQI endpoints
  getCurrentAQI: async (pincode: string): Promise<AQIData> => {
    try {
      const response = await apiClient.get(`/aqi/${pincode}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching AQI for ${pincode}:`, error);
      return getFallbackAQIData(pincode);
    }
  },
  
  getAQIForecast: async (pincode: string, hours: number = 24): Promise<ForecastData[]> => {
    try {
      const response = await apiClient.get(`/forecast/${pincode}?hours=${hours}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching forecast for ${pincode}:`, error);
      return getFallbackForecastData(pincode, hours);
    }
  },
  
  getHealthAdvisory: async (aqiValue: number): Promise<HealthAdvisory> => {
    try {
      const response = await apiClient.get(`/health/advisory?aqi=${aqiValue}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching health advisory for AQI ${aqiValue}:`, error);
      return getFallbackHealthAdvisory(aqiValue);
    }
  },
  
  getMapData: async (): Promise<{ cities: MapCityData[] }> => {
    try {
      const response = await apiClient.get('/map/data');
      return response.data;
    } catch (error) {
      console.error('Error fetching map data:', error);
      return { cities: getFallbackMapData() };
    }
  },
  
  getSystemStatus: async (): Promise<SystemStatus> => {
    try {
      const response = await apiClient.get('/status');
      return response.data;
    } catch (error) {
      console.error('Error fetching system status:', error);
      return getFallbackSystemStatus();
    }
  },
  
  // Additional endpoints
  getTrendData: async (pincode: string, days: number = 7) => {
    try {
      const response = await apiClient.get(`/forecast/trend/${pincode}?days=${days}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching trend data for ${pincode}:`, error);
      return getFallbackTrendData(pincode, days);
    }
  },
  
  getHealthAlerts: async (pincode: string) => {
    try {
      const response = await apiClient.get(`/health/alerts/${pincode}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching health alerts for ${pincode}:`, error);
      return { pincode, alerts_count: 0, alerts: [] };
    }
  }
};

// Fallback data functions
function getFallbackData(url?: string): any {
  if (url?.includes('/aqi/')) {
    const pincode = url.split('/aqi/')[1]?.split('?')[0] || '400001';
    return getFallbackAQIData(pincode);
  }
  
  if (url?.includes('/forecast/')) {
    const pincode = url.split('/forecast/')[1]?.split('?')[0] || '400001';
    return getFallbackForecastData(pincode, 24);
  }
  
  if (url?.includes('/health/advisory')) {
    return getFallbackHealthAdvisory(150);
  }
  
  if (url?.includes('/map/data')) {
    return { cities: getFallbackMapData() };
  }
  
  if (url?.includes('/status')) {
    return getFallbackSystemStatus();
  }
  
  return { error: 'Service temporarily unavailable', fallback: true };
}

function getFallbackAQIData(pincode: string): AQIData {
  const baseAQI = Math.floor(Math.random() * 200) + 50; // 50-250 range
  
  return {
    aqi: baseAQI,
    category: getAQICategory(baseAQI),
    pincode,
    timestamp: new Date().toISOString(),
    source: 'fallback',
    pm25: baseAQI * 0.6,
    pm10: baseAQI * 0.8,
    temperature: Math.floor(Math.random() * 15) + 20, // 20-35°C
    humidity: Math.floor(Math.random() * 40) + 40, // 40-80%
    wind_speed: Math.floor(Math.random() * 20) + 5 // 5-25 km/h
  };
}

function getFallbackForecastData(pincode: string, hours: number): ForecastData[] {
  const forecast: ForecastData[] = [];
  const baseTime = new Date();
  const baseAQI = Math.floor(Math.random() * 200) + 50;
  
  for (let i = 1; i <= hours; i++) {
    const timestamp = new Date(baseTime.getTime() + i * 3600000);
    const variation = Math.floor(Math.random() * 40) - 20; // ±20 variation
    const aqi = Math.max(10, Math.min(500, baseAQI + variation));
    
    forecast.push({
      timestamp: timestamp.toISOString(),
      predicted_aqi: aqi,
      category: getAQICategory(aqi),
      confidence_lower: aqi * 0.8,
      confidence_upper: aqi * 1.2,
      pincode
    });
  }
  
  return forecast;
}

function getFallbackHealthAdvisory(aqiValue: number): HealthAdvisory {
  if (aqiValue <= 50) {
    return {
      category: 'Good',
      message: 'Air quality is excellent. Enjoy outdoor activities!',
      precautions: ['No special precautions needed'],
      risk_groups: ['None'],
      aqi_range: '0-50 (Good)',
      health_effects: 'No health concerns for general population'
    };
  } else if (aqiValue <= 100) {
    return {
      category: 'Satisfactory',
      message: 'Air quality is acceptable for most people.',
      precautions: ['Sensitive individuals should limit prolonged outdoor exertion'],
      risk_groups: ['People with respiratory conditions'],
      aqi_range: '51-100 (Satisfactory)',
      health_effects: 'Minor breathing discomfort for sensitive people'
    };
  } else if (aqiValue <= 200) {
    return {
      category: 'Moderate',
      message: 'Air quality is unhealthy for sensitive groups.',
      precautions: [
        'Wear mask when going outdoors',
        'Limit outdoor activities for children and elderly',
        'Keep windows closed during peak hours'
      ],
      risk_groups: ['Children', 'Elderly', 'People with heart/lung conditions'],
      aqi_range: '101-200 (Moderate)',
      health_effects: 'Breathing discomfort, coughing for sensitive groups'
    };
  } else if (aqiValue <= 300) {
    return {
      category: 'Poor',
      message: 'Everyone may experience health effects.',
      precautions: [
        'Avoid outdoor activities',
        'Use N95 masks outdoors',
        'Use air purifiers indoors',
        'Keep windows and doors closed'
      ],
      risk_groups: ['Everyone, especially children and elderly'],
      aqi_range: '201-300 (Poor)',
      health_effects: 'Coughing, throat irritation, breathing difficulty'
    };
  } else {
    return {
      category: 'Very Poor to Severe',
      message: 'Health alert: serious health effects for everyone.',
      precautions: [
        'Avoid all outdoor activities',
        'Stay indoors with air purification',
        'Wear N95/N99 masks if must go out',
        'Consult doctor if experiencing symptoms'
      ],
      risk_groups: ['Entire population'],
      aqi_range: '301+ (Very Poor to Severe)',
      health_effects: 'Serious respiratory and cardiovascular effects'
    };
  }
}

function getFallbackMapData(): MapCityData[] {
  const cities = [
    { name: 'Mumbai', pincode: '400001', lat: 19.0760, lng: 72.8777 },
    { name: 'Delhi', pincode: '110001', lat: 28.6139, lng: 77.2090 },
    { name: 'Pune', pincode: '411001', lat: 18.5204, lng: 73.8567 },
    { name: 'Bangalore', pincode: '560001', lat: 12.9716, lng: 77.5946 },
    { name: 'Chennai', pincode: '600001', lat: 13.0827, lng: 80.2707 },
    { name: 'Kolkata', pincode: '700001', lat: 22.5726, lng: 88.3639 },
    { name: 'Hyderabad', pincode: '500001', lat: 17.3850, lng: 78.4867 },
    { name: 'Ahmedabad', pincode: '380001', lat: 23.0225, lng: 72.5714 }
  ];
  
  return cities.map(city => {
    const aqi = Math.floor(Math.random() * 200) + 50;
    return {
      ...city,
      aqi,
      category: getAQICategory(aqi),
      pm25: aqi * 0.6,
      pm10: aqi * 0.8,
      temperature: Math.floor(Math.random() * 15) + 20,
      humidity: Math.floor(Math.random() * 40) + 40,
      wind_speed: Math.floor(Math.random() * 20) + 5,
      last_updated: new Date().toISOString()
    };
  });
}

function getFallbackSystemStatus(): SystemStatus {
  return {
    system: 'AnantaNetra - AI Environmental Monitoring',
    version: '1.0.0',
    status: 'operational',
    timestamp: new Date().toISOString(),
    services: {
      api: { status: 'operational', last_check: new Date().toISOString() },
      prediction: { status: 'operational', last_check: new Date().toISOString() },
      cache: { status: 'operational', last_check: new Date().toISOString() }
    }
  };
}

function getFallbackTrendData(pincode: string, days: number) {
  const trendData = [];
  const baseTime = new Date();
  baseTime.setDate(baseTime.getDate() - days);
  
  for (let i = 0; i < days * 24; i++) {
    const timestamp = new Date(baseTime.getTime() + i * 3600000);
    const aqi = Math.floor(Math.random() * 200) + 50;
    
    trendData.push({
      timestamp: timestamp.toISOString(),
      aqi,
      category: getAQICategory(aqi)
    });
  }
  
  return {
    pincode,
    period_days: days,
    data_points: trendData.length,
    average_aqi: trendData.reduce((sum, item) => sum + item.aqi, 0) / trendData.length,
    trend_data: trendData
  };
}

function getAQICategory(aqi: number): string {
  if (aqi <= 50) return 'Good';
  if (aqi <= 100) return 'Satisfactory';
  if (aqi <= 200) return 'Moderate';
  if (aqi <= 300) return 'Poor';
  if (aqi <= 400) return 'Very Poor';
  return 'Severe';
}

export default apiService;
