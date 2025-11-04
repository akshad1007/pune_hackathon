import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Alert,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
} from '@mui/material';
import {
  HealthAndSafety as HealthIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  Masks as MaskIcon,
  DirectionsRun as ExerciseIcon,
  Home as HomeIcon,
  LocalHospital as HospitalIcon,
} from '@mui/icons-material';

import { useAppContext } from '../context/AppContext';
import apiService from '../services/api';
import { AQIData, HealthAdvisory } from '../types';

const HealthAdvisoryPage: React.FC = () => {
  const { selectedPincode } = useAppContext();
  const [currentAQI, setCurrentAQI] = useState<AQIData | null>(null);
  const [healthAdvisory, setHealthAdvisory] = useState<HealthAdvisory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealthData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch current AQI
      const aqiData = await apiService.getCurrentAQI(selectedPincode);
      setCurrentAQI(aqiData);

      // Fetch health advisory
      const advisory = await apiService.getHealthAdvisory(aqiData.aqi);
      setHealthAdvisory(advisory);
    } catch (err) {
      console.error('Error fetching health data:', err);
      setError('Unable to fetch health advisory data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealthData();
  }, [selectedPincode]);

  const getAQIColor = (aqi: number): string => {
    if (aqi <= 50) return '#4caf50';
    if (aqi <= 100) return '#8bc34a';
    if (aqi <= 200) return '#ff9800';
    if (aqi <= 300) return '#f44336';
    if (aqi <= 400) return '#9c27b0';
    return '#8b0000';
  };

  const getSeverityIcon = (aqi: number) => {
    if (aqi <= 50) return <CheckIcon sx={{ color: '#4caf50' }} />;
    if (aqi <= 100) return <InfoIcon sx={{ color: '#8bc34a' }} />;
    if (aqi <= 200) return <WarningIcon sx={{ color: '#ff9800' }} />;
    return <WarningIcon sx={{ color: '#f44336' }} />;
  };

  const getSeverity = (aqi: number): 'success' | 'info' | 'warning' | 'error' => {
    if (aqi <= 50) return 'success';
    if (aqi <= 100) return 'info';
    if (aqi <= 200) return 'warning';
    return 'error';
  };

  const generalPrecautionsByCategory = {
    'Good': {
      icon: <CheckIcon />,
      color: '#4caf50',
      precautions: [
        'Enjoy outdoor activities',
        'No special precautions needed',
        'Ideal time for exercise and outdoor sports',
        'Open windows for fresh air'
      ]
    },
    'Satisfactory': {
      icon: <InfoIcon />,
      color: '#8bc34a',
      precautions: [
        'Generally safe for outdoor activities',
        'Sensitive individuals may consider limiting prolonged outdoor exertion',
        'Monitor air quality if you have respiratory conditions'
      ]
    },
    'Moderate': {
      icon: <WarningIcon />,
      color: '#ff9800',
      precautions: [
        'Limit outdoor activities for sensitive groups',
        'Wear N95 mask when going outdoors',
        'Keep windows closed during peak pollution hours (7-10 AM, 7-10 PM)',
        'Use air purifiers indoors if available'
      ]
    },
    'Poor': {
      icon: <WarningIcon />,
      color: '#f44336',
      precautions: [
        'Avoid outdoor activities, especially for children and elderly',
        'Mandatory use of N95/N99 masks outdoors',
        'Keep all windows and doors closed',
        'Use air purifiers and indoor plants',
        'Limit vehicle use to reduce emissions'
      ]
    },
    'Very Poor': {
      icon: <WarningIcon />,
      color: '#9c27b0',
      precautions: [
        'Stay indoors as much as possible',
        'Avoid all outdoor physical activities',
        'Use N95/N99 masks even for short outdoor exposure',
        'Seal windows and doors to prevent pollution entry',
        'Consider relocating temporarily if possible'
      ]
    },
    'Severe': {
      icon: <WarningIcon />,
      color: '#8b0000',
      precautions: [
        'Emergency measures: Stay indoors at all times',
        'Use industrial-grade air purifiers',
        'Wear N99/P100 masks for any outdoor exposure',
        'Seek medical attention if experiencing breathing difficulties',
        'Close all schools and offices if possible'
      ]
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading health advisory for {selectedPincode}...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        Health Advisory & Recommendations
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Current Status Card */}
        <Grid item xs={12} lg={4}>
          <Card 
            sx={{ 
              background: currentAQI ? `linear-gradient(135deg, ${getAQIColor(currentAQI.aqi)}20 0%, ${getAQIColor(currentAQI.aqi)}40 100%)` : 'background.paper',
              border: `2px solid ${currentAQI ? getAQIColor(currentAQI.aqi) : 'divider'}`,
            }}
          >
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <HealthIcon fontSize="large" />
                {currentAQI && getSeverityIcon(currentAQI.aqi)}
              </Box>
              <Typography variant="h3" component="div" fontWeight="bold">
                {currentAQI?.aqi || '--'}
              </Typography>
              <Chip 
                label={currentAQI?.category || 'Loading...'} 
                sx={{ 
                  backgroundColor: currentAQI ? getAQIColor(currentAQI.aqi) : 'grey.300',
                  color: 'white',
                  fontWeight: 'bold',
                  mb: 2
                }}
              />
              <Typography variant="body1" color="text.secondary">
                Current Air Quality Index
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Location: {selectedPincode}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Health Impact Overview */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Health Impact Assessment
              </Typography>
              {healthAdvisory && (
                <Alert severity={getSeverity(currentAQI?.aqi || 0)} sx={{ mb: 2 }}>
                  <Typography variant="body1" fontWeight="bold">
                    {healthAdvisory.message}
                  </Typography>
                </Alert>
              )}
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    AQI Range
                  </Typography>
                  <Typography variant="body1">{healthAdvisory?.aqi_range || 'Loading...'}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Health Effects
                  </Typography>
                  <Typography variant="body1">{healthAdvisory?.health_effects || 'Loading...'}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Detailed Recommendations */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detailed Health Recommendations
              </Typography>
              
              {/* AI-Generated Precautions */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box display="flex" alignItems="center">
                    <MaskIcon sx={{ mr: 1 }} />
                    <Typography variant="subtitle1" fontWeight="bold">
                      Immediate Precautions
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <List>
                    {healthAdvisory?.precautions.map((precaution, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <CheckIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText primary={precaution} />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>

              {/* Risk Groups */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box display="flex" alignItems="center">
                    <WarningIcon sx={{ mr: 1 }} />
                    <Typography variant="subtitle1" fontWeight="bold">
                      At-Risk Groups
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <List>
                    {healthAdvisory?.risk_groups.map((group, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <WarningIcon color="warning" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={group}
                          secondary="Should take extra precautions and limit outdoor exposure"
                        />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>

              {/* Outdoor Activities */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box display="flex" alignItems="center">
                    <ExerciseIcon sx={{ mr: 1 }} />
                    <Typography variant="subtitle1" fontWeight="bold">
                      Outdoor Activity Guidelines
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  {currentAQI && (
                    <List>
                      {generalPrecautionsByCategory[currentAQI.category as keyof typeof generalPrecautionsByCategory]?.precautions.map((item, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            {generalPrecautionsByCategory[currentAQI.category as keyof typeof generalPrecautionsByCategory].icon}
                          </ListItemIcon>
                          <ListItemText primary={item} />
                        </ListItem>
                      ))}
                    </List>
                  )}
                </AccordionDetails>
              </Accordion>

              {/* Indoor Safety */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box display="flex" alignItems="center">
                    <HomeIcon sx={{ mr: 1 }} />
                    <Typography variant="subtitle1" fontWeight="bold">
                      Indoor Air Quality Tips
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <List>
                    <ListItem>
                      <ListItemIcon><CheckIcon color="primary" /></ListItemIcon>
                      <ListItemText 
                        primary="Use air purifiers with HEPA filters"
                        secondary="Essential for PM2.5 and PM10 filtration"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><CheckIcon color="primary" /></ListItemIcon>
                      <ListItemText 
                        primary="Add indoor plants"
                        secondary="Spider plants, peace lilies, and snake plants help purify air"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><CheckIcon color="primary" /></ListItemIcon>
                      <ListItemText 
                        primary="Seal gaps in windows and doors"
                        secondary="Prevent outdoor pollutants from entering"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><CheckIcon color="primary" /></ListItemIcon>
                      <ListItemText 
                        primary="Avoid indoor pollution sources"
                        secondary="No smoking, minimize cooking smoke, avoid strong chemicals"
                      />
                    </ListItem>
                  </List>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        </Grid>

        {/* Emergency Information */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <HospitalIcon sx={{ mr: 1 }} color="error" />
                <Typography variant="h6" color="error">
                  When to Seek Medical Help
                </Typography>
              </Box>
              <List>
                <ListItem>
                  <ListItemIcon><WarningIcon color="error" /></ListItemIcon>
                  <ListItemText 
                    primary="Difficulty breathing or shortness of breath"
                    secondary="Especially in children and elderly"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><WarningIcon color="error" /></ListItemIcon>
                  <ListItemText 
                    primary="Persistent cough or throat irritation"
                    secondary="Lasting more than 2-3 days"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><WarningIcon color="error" /></ListItemIcon>
                  <ListItemText 
                    primary="Chest pain or discomfort"
                    secondary="May indicate cardiovascular stress"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><WarningIcon color="error" /></ListItemIcon>
                  <ListItemText 
                    primary="Severe headaches or dizziness"
                    secondary="Could be related to poor air quality"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Air Quality Scale Reference */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AQI Scale Reference
              </Typography>
              <List dense>
                {[
                  { range: '0-50', category: 'Good', color: '#4caf50', description: 'Minimal impact' },
                  { range: '51-100', category: 'Satisfactory', color: '#8bc34a', description: 'Minor breathing discomfort for sensitive people' },
                  { range: '101-200', category: 'Moderate', color: '#ff9800', description: 'Breathing discomfort for people with lung/heart disease' },
                  { range: '201-300', category: 'Poor', color: '#f44336', description: 'Breathing discomfort for most people' },
                  { range: '301-400', category: 'Very Poor', color: '#9c27b0', description: 'Respiratory illness on prolonged exposure' },
                  { range: '401-500', category: 'Severe', color: '#8b0000', description: 'Affects healthy people and seriously impacts those with existing diseases' },
                ].map((item, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <Box
                        sx={{
                          width: 20,
                          height: 20,
                          backgroundColor: item.color,
                          borderRadius: 1,
                        }}
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary={`${item.range} - ${item.category}`}
                      secondary={item.description}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Refresh Button */}
        <Grid item xs={12}>
          <Box textAlign="center">
            <Button
              variant="contained"
              onClick={fetchHealthData}
              disabled={loading}
              size="large"
            >
              {loading ? 'Updating...' : 'Refresh Health Advisory'}
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HealthAdvisoryPage;
