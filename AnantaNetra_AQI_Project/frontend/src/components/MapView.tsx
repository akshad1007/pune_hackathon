import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Alert,
  CircularProgress,
  Fab,
  Tooltip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  MyLocation as LocationIcon,
} from '@mui/icons-material';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

import { useAppContext } from '../context/AppContext';
import apiService from '../services/api';
import { MapCityData } from '../types';

// Fix for default markers in React Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom AQI marker icons
const createAQIIcon = (aqi: number) => {
  const color = getAQIColor(aqi);
  const size = aqi > 200 ? 25 : 20;
  
  return L.divIcon({
    html: `
      <div style="
        background-color: ${color};
        color: white;
        border-radius: 50%;
        width: ${size}px;
        height: ${size}px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 12px;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      ">
        ${aqi}
      </div>
    `,
    className: 'aqi-marker',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2],
  });
};

function getAQIColor(aqi: number): string {
  if (aqi <= 50) return '#4caf50';
  if (aqi <= 100) return '#8bc34a';
  if (aqi <= 200) return '#ff9800';
  if (aqi <= 300) return '#f44336';
  if (aqi <= 400) return '#9c27b0';
  return '#8b0000';
}

// Component to handle map interactions
const MapController: React.FC<{ 
  selectedPincode: string; 
  cities: MapCityData[];
  onCitySelect: (pincode: string) => void;
}> = ({ selectedPincode, cities, onCitySelect }) => {
  const map = useMap();

  // Center map on selected city
  useEffect(() => {
    const selectedCity = cities.find(city => city.pincode === selectedPincode);
    if (selectedCity) {
      map.setView([selectedCity.lat, selectedCity.lng], 12);
    }
  }, [selectedPincode, cities, map]);

  return null;
};

const MapView: React.FC = () => {
  const { selectedPincode, setSelectedPincode } = useAppContext();
  const [mapData, setMapData] = useState<MapCityData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCity, setSelectedCity] = useState<MapCityData | null>(null);

  const fetchMapData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.getMapData();
      setMapData(response.cities);
      
      // Set selected city based on current pincode
      const currentCity = response.cities.find(city => city.pincode === selectedPincode);
      if (currentCity) {
        setSelectedCity(currentCity);
      }
    } catch (err) {
      console.error('Error fetching map data:', err);
      setError('Unable to load map data. Using fallback locations.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMapData();
  }, []);

  const handleCitySelect = (city: MapCityData) => {
    setSelectedCity(city);
    setSelectedPincode(city.pincode);
  };

  const centerIndia = [20.5937, 78.9629]; // Geographic center of India

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading map data...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        India Air Quality Map
      </Typography>

      {error && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Map Container */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent sx={{ p: 0 }}>
              <Box sx={{ height: '600px', position: 'relative' }}>
                <MapContainer
                  center={centerIndia as [number, number]}
                  zoom={5}
                  style={{ height: '100%', width: '100%' }}
                  scrollWheelZoom={true}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  
                  <MapController 
                    selectedPincode={selectedPincode} 
                    cities={mapData}
                    onCitySelect={setSelectedPincode}
                  />
                  
                  {mapData.map((city) => (
                    <Marker
                      key={city.pincode}
                      position={[city.lat, city.lng]}
                      icon={createAQIIcon(city.aqi)}
                      eventHandlers={{
                        click: () => handleCitySelect(city),
                      }}
                    >
                      <Popup>
                        <Box sx={{ minWidth: 200 }}>
                          <Typography variant="h6" fontWeight="bold">
                            {city.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            Pincode: {city.pincode}
                          </Typography>
                          
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="h5" fontWeight="bold" sx={{ mr: 1 }}>
                              {city.aqi}
                            </Typography>
                            <Chip
                              label={city.category}
                              size="small"
                              sx={{
                                backgroundColor: getAQIColor(city.aqi),
                                color: 'white',
                                fontWeight: 'bold',
                              }}
                            />
                          </Box>
                          
                          <Grid container spacing={1} sx={{ mt: 1 }}>
                            <Grid item xs={6}>
                              <Typography variant="body2">
                                <strong>PM2.5:</strong> {city.pm25?.toFixed(1)} μg/m³
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="body2">
                                <strong>PM10:</strong> {city.pm10?.toFixed(1)} μg/m³
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="body2">
                                <strong>Temp:</strong> {city.temperature}°C
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="body2">
                                <strong>Humidity:</strong> {city.humidity}%
                              </Typography>
                            </Grid>
                          </Grid>
                          
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                            Updated: {new Date(city.last_updated).toLocaleTimeString()}
                          </Typography>
                        </Box>
                      </Popup>
                    </Marker>
                  ))}
                </MapContainer>
                
                <Fab
                  color="primary"
                  size="small"
                  sx={{ position: 'absolute', top: 16, right: 16, zIndex: 1000 }}
                  onClick={fetchMapData}
                  disabled={loading}
                >
                  <RefreshIcon />
                </Fab>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* City Details & Legend */}
        <Grid item xs={12} lg={4}>
          {/* Selected City Details */}
          {selectedCity && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Selected Location
                </Typography>
                
                <Box 
                  sx={{ 
                    p: 2, 
                    borderRadius: 2,
                    background: `linear-gradient(135deg, ${getAQIColor(selectedCity.aqi)}20 0%, ${getAQIColor(selectedCity.aqi)}40 100%)`,
                    border: `2px solid ${getAQIColor(selectedCity.aqi)}`,
                    mb: 2
                  }}
                >
                  <Typography variant="h5" fontWeight="bold">
                    {selectedCity.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {selectedCity.pincode} • Lat: {selectedCity.lat.toFixed(4)}, Lng: {selectedCity.lng.toFixed(4)}
                  </Typography>
                  
                  <Box display="flex" alignItems="center" mb={2}>
                    <Typography variant="h3" fontWeight="bold" sx={{ mr: 2 }}>
                      {selectedCity.aqi}
                    </Typography>
                    <Chip
                      label={selectedCity.category}
                      sx={{
                        backgroundColor: getAQIColor(selectedCity.aqi),
                        color: 'white',
                        fontWeight: 'bold',
                      }}
                    />
                  </Box>
                  
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="body2">
                        <strong>PM2.5:</strong> {selectedCity.pm25?.toFixed(1)} μg/m³
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2">
                        <strong>PM10:</strong> {selectedCity.pm10?.toFixed(1)} μg/m³
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2">
                        <strong>Temperature:</strong> {selectedCity.temperature}°C
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2">
                        <strong>Humidity:</strong> {selectedCity.humidity}%
                      </Typography>
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="body2">
                        <strong>Wind Speed:</strong> {selectedCity.wind_speed} km/h
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
                
                <Typography variant="caption" color="text.secondary">
                  Last updated: {new Date(selectedCity.last_updated).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* AQI Legend */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AQI Scale Legend
              </Typography>
              
              {[
                { range: '0-50', category: 'Good', color: '#4caf50' },
                { range: '51-100', category: 'Satisfactory', color: '#8bc34a' },
                { range: '101-200', category: 'Moderate', color: '#ff9800' },
                { range: '201-300', category: 'Poor', color: '#f44336' },
                { range: '301-400', category: 'Very Poor', color: '#9c27b0' },
                { range: '401-500', category: 'Severe', color: '#8b0000' },
              ].map((item, index) => (
                <Box key={index} display="flex" alignItems="center" mb={1}>
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      backgroundColor: item.color,
                      borderRadius: '50%',
                      mr: 1,
                      border: '2px solid white',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                    }}
                  />
                  <Typography variant="body2">
                    {item.range} - {item.category}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>

          {/* Map Instructions */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Map Instructions
              </Typography>
              
              <Box component="ul" sx={{ pl: 2, m: 0 }}>
                <li>
                  <Typography variant="body2">
                    Click on city markers to view detailed AQI information
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    Marker colors represent AQI categories (green = good, red = poor)
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    Use mouse wheel to zoom in/out
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    Drag to pan across different regions
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    Click refresh button to update real-time data
                  </Typography>
                </li>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Summary Statistics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                National Overview
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Cities Monitored
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {mapData.length}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Average AQI
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {mapData.length > 0 ? Math.round(mapData.reduce((sum, city) => sum + city.aqi, 0) / mapData.length) : '--'}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Cities with Poor AQI
                  </Typography>
                  <Typography variant="h5" fontWeight="bold" color="error">
                    {mapData.filter(city => city.aqi > 200).length}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">
                    Cities with Good AQI
                  </Typography>
                  <Typography variant="h5" fontWeight="bold" color="success.main">
                    {mapData.filter(city => city.aqi <= 100).length}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MapView;
