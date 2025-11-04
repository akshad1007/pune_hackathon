import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Container, AppBar, Toolbar, Typography, Switch, FormControlLabel } from '@mui/material';
import { Eco as EcoIcon } from '@mui/icons-material';

import Dashboard from './components/Dashboard';
import MapView from './components/MapView';
import Search from './components/Search';
import HealthAdvisory from './components/HealthAdvisory';
import Navigation from './components/Navigation';
import ErrorBoundary from './components/ErrorBoundary';
import LoadingScreen from './components/LoadingScreen';
import AppContext from './context/AppContext';

import { AppContextType } from './types';
import './App.css';

const App: React.FC = () => {
  const [selectedPincode, setSelectedPincode] = useState<string>('400001'); // Default to Mumbai
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [systemReady, setSystemReady] = useState<boolean>(false);

  // Initialize app
  useEffect(() => {
    const initializeApp = async () => {
      setIsLoading(true);
      
      // Simulate initialization time
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Load saved preferences
      const savedTheme = localStorage.getItem('ananta-theme') as 'light' | 'dark';
      if (savedTheme) {
        setTheme(savedTheme);
      }
      
      const savedPincode = localStorage.getItem('ananta-pincode');
      if (savedPincode) {
        setSelectedPincode(savedPincode);
      }
      
      setSystemReady(true);
      setIsLoading(false);
    };
    
    initializeApp();
  }, []);

  // Save preferences when they change
  useEffect(() => {
    localStorage.setItem('ananta-theme', theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('ananta-pincode', selectedPincode);
  }, [selectedPincode]);

  // Create Material-UI theme
  const muiTheme = createTheme({
    palette: {
      mode: theme,
      primary: {
        main: theme === 'light' ? '#2e7d32' : '#4caf50',
        light: '#60ad5e',
        dark: '#1b5e20',
      },
      secondary: {
        main: theme === 'light' ? '#1976d2' : '#42a5f5',
        light: '#63a4ff',
        dark: '#004ba0',
      },
      error: {
        main: '#d32f2f',
        light: '#ef5350',
        dark: '#c62828',
      },
      warning: {
        main: '#ed6c02',
        light: '#ff9800',
        dark: '#e65100',
      },
      success: {
        main: '#2e7d32',
        light: '#4caf50',
        dark: '#1b5e20',
      },
      background: {
        default: theme === 'light' ? '#f5f5f5' : '#121212',
        paper: theme === 'light' ? '#ffffff' : '#1e1e1e',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontSize: '2.5rem',
        fontWeight: 600,
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 500,
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 500,
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 500,
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 500,
      },
      h6: {
        fontSize: '1rem',
        fontWeight: 500,
      },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            boxShadow: theme === 'light' 
              ? '0 2px 8px rgba(0,0,0,0.1)' 
              : '0 2px 8px rgba(255,255,255,0.1)',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            borderRadius: 8,
          },
        },
      },
    },
  });

  // Context value
  const contextValue: AppContextType = {
    selectedPincode,
    setSelectedPincode,
    theme,
    setTheme,
    isLoading,
    setIsLoading,
  };

  const handleThemeToggle = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  if (!systemReady) {
    return <LoadingScreen />;
  }

  return (
    <ThemeProvider theme={muiTheme}>
      <CssBaseline />
      <AppContext.Provider value={contextValue}>
        <ErrorBoundary>
          <Router>
            <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
              {/* App Header */}
              <AppBar position="static" elevation={1}>
                <Toolbar>
                  <EcoIcon sx={{ mr: 2 }} />
                  <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                    AnantaNetra - AI Environmental Monitoring
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={theme === 'dark'}
                        onChange={handleThemeToggle}
                        color="default"
                      />
                    }
                    label={theme === 'dark' ? '🌙' : '☀️'}
                    sx={{ ml: 2 }}
                  />
                </Toolbar>
              </AppBar>

              {/* Navigation */}
              <Navigation />

              {/* Main Content */}
              <Box
                component="main"
                sx={{
                  flexGrow: 1,
                  backgroundColor: 'background.default',
                  minHeight: 'calc(100vh - 128px)', // Subtract header and nav height
                }}
              >
                <Container maxWidth="xl" sx={{ py: 3 }}>
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/map" element={<MapView />} />
                    <Route path="/search" element={<Search />} />
                    <Route path="/health" element={<HealthAdvisory />} />
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                  </Routes>
                </Container>
              </Box>

              {/* Footer */}
              <Box
                component="footer"
                sx={{
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  py: 2,
                  textAlign: 'center',
                }}
              >
                <Typography variant="body2">
                  © 2025 AnantaNetra - AI-Powered Environmental Monitoring for India
                </Typography>
                <Typography variant="caption">
                  Protecting 2M+ lives with real-time AQI monitoring & AI-powered health insights
                </Typography>
              </Box>
            </Box>
          </Router>
        </ErrorBoundary>
      </AppContext.Provider>
    </ThemeProvider>
  );
};

export default App;
