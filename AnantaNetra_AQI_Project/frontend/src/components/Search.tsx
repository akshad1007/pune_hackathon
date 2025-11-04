import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  CircularProgress,
  Autocomplete,
} from '@mui/material';
import {
  Search as SearchIcon,
  LocationOn as LocationIcon,
  Air as AirIcon,
  History as HistoryIcon,
} from '@mui/icons-material';

import { useAppContext } from '../context/AppContext';
import apiService from '../services/api';
import { AQIData } from '../types';

// Common Indian pincodes for autocomplete
const popularPincodes = [
  { pincode: '110001', city: 'New Delhi', state: 'Delhi' },
  { pincode: '400001', city: 'Mumbai', state: 'Maharashtra' },
  { pincode: '560001', city: 'Bangalore', state: 'Karnataka' },
  { pincode: '600001', city: 'Chennai', state: 'Tamil Nadu' },
  { pincode: '700001', city: 'Kolkata', state: 'West Bengal' },
  { pincode: '411001', city: 'Pune', state: 'Maharashtra' },
  { pincode: '500001', city: 'Hyderabad', state: 'Telangana' },
  { pincode: '380001', city: 'Ahmedabad', state: 'Gujarat' },
  { pincode: '302001', city: 'Jaipur', state: 'Rajasthan' },
  { pincode: '800001', city: 'Patna', state: 'Bihar' },
  { pincode: '226001', city: 'Lucknow', state: 'Uttar Pradesh' },
  { pincode: '160001', city: 'Chandigarh', state: 'Chandigarh' },
  { pincode: '682001', city: 'Kochi', state: 'Kerala' },
  { pincode: '751001', city: 'Bhubaneswar', state: 'Odisha' },
  { pincode: '780001', city: 'Guwahati', state: 'Assam' },
];

interface SearchResult {
  pincode: string;
  aqiData: AQIData;
  timestamp: Date;
}

const Search: React.FC = () => {
  const { selectedPincode, setSelectedPincode } = useAppContext();
  const [searchPincode, setSearchPincode] = useState(selectedPincode);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load search history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('ananta-search-history');
    if (savedHistory) {
      try {
        setSearchHistory(JSON.parse(savedHistory));
      } catch (err) {
        console.error('Error loading search history:', err);
      }
    }
  }, []);

  // Save search history to localStorage
  const saveSearchHistory = (pincode: string) => {
    const updatedHistory = [pincode, ...searchHistory.filter(p => p !== pincode)].slice(0, 10);
    setSearchHistory(updatedHistory);
    localStorage.setItem('ananta-search-history', JSON.stringify(updatedHistory));
  };

  const handleSearch = async () => {
    if (!searchPincode || searchPincode.length !== 6) {
      setError('Please enter a valid 6-digit pincode');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const aqiData = await apiService.getCurrentAQI(searchPincode);
      
      const newResult: SearchResult = {
        pincode: searchPincode,
        aqiData,
        timestamp: new Date(),
      };

      setSearchResults(prev => [newResult, ...prev.slice(0, 4)]); // Keep last 5 results
      saveSearchHistory(searchPincode);
      setSelectedPincode(searchPincode);
    } catch (err) {
      console.error('Search error:', err);
      setError('Unable to fetch AQI data for this pincode. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePincodeSelect = (pincode: string) => {
    setSearchPincode(pincode);
    setSelectedPincode(pincode);
  };

  const getAQIColor = (aqi: number): string => {
    if (aqi <= 50) return '#4caf50';
    if (aqi <= 100) return '#8bc34a';
    if (aqi <= 200) return '#ff9800';
    if (aqi <= 300) return '#f44336';
    if (aqi <= 400) return '#9c27b0';
    return '#8b0000';
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        Search Air Quality by Location
      </Typography>

      <Grid container spacing={3}>
        {/* Search Section */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Enter Location Details
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  label="Pincode"
                  value={searchPincode}
                  onChange={(e) => setSearchPincode(e.target.value)}
                  placeholder="Enter 6-digit pincode"
                  inputProps={{ maxLength: 6, pattern: '[0-9]*' }}
                  helperText="Enter Indian pincode (e.g., 400001 for Mumbai)"
                  sx={{ mb: 2 }}
                />
                
                <Button
                  fullWidth
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
                  onClick={handleSearch}
                  disabled={loading || !searchPincode}
                  size="large"
                >
                  {loading ? 'Searching...' : 'Get Air Quality Data'}
                </Button>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              {/* Popular Locations */}
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                Popular Locations
              </Typography>
              <Autocomplete
                options={popularPincodes}
                getOptionLabel={(option) => `${option.pincode} - ${option.city}, ${option.state}`}
                renderInput={(params) => (
                  <TextField {...params} label="Select popular location" placeholder="Choose a city" />
                )}
                onChange={(_, value) => {
                  if (value) {
                    handlePincodeSelect(value.pincode);
                  }
                }}
                renderOption={(props, option) => (
                  <Box component="li" {...props}>
                    <LocationIcon sx={{ mr: 1, color: 'text.secondary' }} />
                    <Box>
                      <Typography variant="body1">{option.city}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {option.pincode} - {option.state}
                      </Typography>
                    </Box>
                  </Box>
                )}
              />
            </CardContent>
          </Card>

          {/* Search History */}
          {searchHistory.length > 0 && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <HistoryIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Recent Searches</Typography>
                </Box>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  {searchHistory.map((pincode, index) => (
                    <Chip
                      key={index}
                      label={pincode}
                      onClick={() => handlePincodeSelect(pincode)}
                      variant="outlined"
                      clickable
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Search Results */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Search Results
              </Typography>
              
              {searchResults.length === 0 ? (
                <Box textAlign="center" py={4}>
                  <AirIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    Search for a location to view air quality data
                  </Typography>
                </Box>
              ) : (
                <List>
                  {searchResults.map((result, index) => {
                    const cityInfo = popularPincodes.find(p => p.pincode === result.pincode);
                    return (
                      <ListItem
                        key={index}
                        sx={{
                          mb: 2,
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 2,
                          backgroundColor: index === 0 ? 'action.selected' : 'background.paper',
                        }}
                      >
                        <ListItemIcon>
                          <Box
                            sx={{
                              width: 48,
                              height: 48,
                              borderRadius: '50%',
                              backgroundColor: getAQIColor(result.aqiData.aqi),
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white',
                              fontWeight: 'bold',
                            }}
                          >
                            {result.aqiData.aqi}
                          </Box>
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box>
                              <Typography variant="subtitle1" fontWeight="bold">
                                {cityInfo ? `${cityInfo.city}, ${cityInfo.state}` : `Pincode ${result.pincode}`}
                              </Typography>
                              <Chip
                                label={result.aqiData.category}
                                size="small"
                                sx={{
                                  backgroundColor: getAQIColor(result.aqiData.aqi),
                                  color: 'white',
                                  fontWeight: 'bold',
                                  mt: 0.5,
                                }}
                              />
                            </Box>
                          }
                          secondary={
                            <Box sx={{ mt: 1 }}>
                              <Typography variant="body2" color="text.secondary">
                                AQI: {result.aqiData.aqi} • PM2.5: {result.aqiData.pm25?.toFixed(1) || 'N/A'} μg/m³
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Searched: {result.timestamp.toLocaleString()}
                              </Typography>
                            </Box>
                          }
                        />
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={() => setSelectedPincode(result.pincode)}
                        >
                          View Details
                        </Button>
                      </ListItem>
                    );
                  })}
                </List>
              )}
            </CardContent>
          </Card>

          {/* Location Tips */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Search Tips
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Use 6-digit Indian pincodes"
                    secondary="Example: 400001 for Mumbai, 110001 for New Delhi"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Popular cities are pre-loaded"
                    secondary="Select from the dropdown for quick access"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Recent searches are saved"
                    secondary="Click on recent search chips for quick access"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Search;
