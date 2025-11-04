import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  Alert,
  CircularProgress,
  Refresh as RefreshIcon,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Air as AirIcon,
  Thermostat as TempIcon,
  Water as HumidityIcon,
  Air as WindIcon,
  TrendingUp as TrendIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer } from 'recharts';

import { useAppContext } from '../context/AppContext';
import apiService from '../services/api';
import { AQIData, ForecastData, HealthAdvisory, ChartDataPoint } from '../types';

const Dashboard: React.FC = () => {
  const { selectedPincode, setIsLoading } = useAppContext();
  const [currentAQI, setCurrentAQI] = useState<AQIData | null>(null);
  const [forecast, setForecast] = useState<ForecastData[]>([]);
  const [healthAdvisory, setHealthAdvisory] = useState<HealthAdvisory | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      setIsLoading(true);

      // Fetch current AQI data
      const aqiData = await apiService.getCurrentAQI(selectedPincode);
      setCurrentAQI(aqiData);

      // Fetch health advisory based on current AQI
      if (aqiData.aqi) {
        const advisory = await apiService.getHealthAdvisory(aqiData.aqi);
        setHealthAdvisory(advisory);
      }

      // Fetch 24-hour forecast
      const forecastData = await apiService.getAQIForecast(selectedPincode, 24);
      setForecast(forecastData);

      // Prepare chart data
      const chartPoints: ChartDataPoint[] = [
        {
          timestamp: aqiData.timestamp,
          aqi: aqiData.aqi,
          category: aqiData.category,
          hour: 'Now',
        },
        ...forecastData.slice(0, 12).map((item, index) => ({
          timestamp: item.timestamp,
          aqi: item.predicted_aqi,
          category: item.category,
          hour: `+${index + 1}h`,
        })),
      ];
      setChartData(chartPoints);

      setLastUpdated(new Date());
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Unable to fetch real-time data. Using fallback data.');
    } finally {
      setLoading(false);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, [selectedPincode]);

  const getAQIColor = (aqi: number): string => {
    if (aqi <= 50) return '#4caf50'; // Good - Green
    if (aqi <= 100) return '#8bc34a'; // Satisfactory - Light Green
    if (aqi <= 200) return '#ff9800'; // Moderate - Orange
    if (aqi <= 300) return '#f44336'; // Poor - Red
    if (aqi <= 400) return '#9c27b0'; // Very Poor - Purple
    return '#8b0000'; // Severe - Dark Red
  };

  const getAQIGradient = (aqi: number): string => {
    const color = getAQIColor(aqi);
    return `linear-gradient(135deg, ${color}20 0%, ${color}40 100%)`;
  };

  if (loading && !currentAQI) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading dashboard data for {selectedPincode}...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" fontWeight="bold">
          Air Quality Dashboard
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          {lastUpdated && (
            <Typography variant="body2" color="text.secondary">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </Typography>
          )}
          <Tooltip title="Refresh data">
            <IconButton onClick={fetchDashboardData} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Current AQI Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card 
            sx={{ 
              background: currentAQI ? getAQIGradient(currentAQI.aqi) : 'background.paper',
              border: `2px solid ${currentAQI ? getAQIColor(currentAQI.aqi) : 'divider'}`,
            }}
          >
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
                <AirIcon fontSize="large" />
                <Chip 
                  label={currentAQI?.category || 'Loading...'} 
                  sx={{ 
                    backgroundColor: currentAQI ? getAQIColor(currentAQI.aqi) : 'grey.300',
                    color: 'white',
                    fontWeight: 'bold'
                  }}
                />
              </Box>
              <Typography variant="h3" component="div" fontWeight="bold">
                {currentAQI?.aqi || '--'}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Air Quality Index
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Pincode: {selectedPincode}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Environmental Parameters */}
        <Grid item xs={12} md={6} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Environmental Parameters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                      <AirIcon color="primary" />
                    </Box>
                    <Typography variant="h6">{currentAQI?.pm25?.toFixed(1) || '--'}</Typography>
                    <Typography variant="body2" color="text.secondary">PM2.5 (μg/m³)</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                      <TempIcon color="primary" />
                    </Box>
                    <Typography variant="h6">{currentAQI?.temperature || '--'}°C</Typography>
                    <Typography variant="body2" color="text.secondary">Temperature</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                      <HumidityIcon color="primary" />
                    </Box>
                    <Typography variant="h6">{currentAQI?.humidity || '--'}%</Typography>
                    <Typography variant="body2" color="text.secondary">Humidity</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                      <WindIcon color="primary" />
                    </Box>
                    <Typography variant="h6">{currentAQI?.wind_speed || '--'}</Typography>
                    <Typography variant="body2" color="text.secondary">Wind (km/h)</Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* AQI Trend Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <TrendIcon sx={{ mr: 1 }} />
                <Typography variant="h6">
                  24-Hour AQI Forecast
                </Typography>
              </Box>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="hour" 
                      tick={{ fontSize: 12 }}
                    />
                    <YAxis 
                      domain={[0, 500]}
                      tick={{ fontSize: 12 }}
                    />
                    <ChartTooltip 
                      formatter={(value: number, name: string) => [
                        `${value} (${chartData.find(d => d.aqi === value)?.category || ''})`,
                        'AQI'
                      ]}
                      labelFormatter={(label) => `Time: ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="aqi" 
                      stroke="#2196f3" 
                      strokeWidth={3}
                      dot={{ fill: '#2196f3', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, stroke: '#2196f3', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Health Advisory */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <WarningIcon sx={{ mr: 1 }} color="warning" />
                <Typography variant="h6">
                  Health Advisory
                </Typography>
              </Box>
              {healthAdvisory ? (
                <Box>
                  <Chip 
                    label={healthAdvisory.category}
                    sx={{ 
                      backgroundColor: currentAQI ? getAQIColor(currentAQI.aqi) : 'grey.300',
                      color: 'white',
                      fontWeight: 'bold',
                      mb: 2
                    }}
                  />
                  <Typography variant="body1" gutterBottom>
                    {healthAdvisory.message}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <strong>AQI Range:</strong> {healthAdvisory.aqi_range}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <strong>Health Effects:</strong> {healthAdvisory.health_effects}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Risk Groups:</strong> {healthAdvisory.risk_groups.join(', ')}
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Loading health advisory...
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Stats */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">Current Status</Typography>
                  <Typography variant="h6" color={currentAQI ? getAQIColor(currentAQI.aqi) : 'text.primary'}>
                    {currentAQI?.category || 'Loading...'}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">Data Source</Typography>
                  <Typography variant="body1">{currentAQI?.source || 'API'}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">PM10 Level</Typography>
                  <Typography variant="body1">{currentAQI?.pm10?.toFixed(1) || '--'} μg/m³</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">Forecast Points</Typography>
                  <Typography variant="body1">{forecast.length} hours</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
