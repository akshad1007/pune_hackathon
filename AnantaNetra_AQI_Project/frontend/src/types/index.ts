// API Response Types
export interface AQIData {
  aqi: number;
  category: string;
  pincode: string;
  timestamp: string;
  source: string;
  pm25?: number;
  pm10?: number;
  temperature?: number;
  humidity?: number;
  wind_speed?: number;
}

export interface ForecastData {
  timestamp: string;
  predicted_aqi: number;
  category: string;
  confidence_lower?: number;
  confidence_upper?: number;
  pincode: string;
}

export interface HealthAdvisory {
  category: string;
  message: string;
  precautions: string[];
  risk_groups: string[];
  aqi_range: string;
  health_effects: string;
}

export interface MapCityData {
  name: string;
  pincode: string;
  lat: number;
  lng: number;
  aqi: number;
  category: string;
  pm25: number;
  pm10: number;
  temperature: number;
  humidity: number;
  wind_speed: number;
  last_updated: string;
}

export interface SystemStatus {
  system: string;
  version: string;
  status: string;
  timestamp: string;
  services: Record<string, ServiceStatus>;
}

export interface ServiceStatus {
  status: string;
  last_check: string;
  response_time_ms?: number;
  error?: string;
}

// UI Component Types
export interface AppContextType {
  selectedPincode: string;
  setSelectedPincode: (pincode: string) => void;
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export interface ChartDataPoint {
  timestamp: string;
  aqi: number;
  category: string;
  hour?: string;
  date?: string;
}

// Error Types
export interface APIError {
  message: string;
  status?: number;
  fallback?: boolean;
}
