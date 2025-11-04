import React from 'react';
import { Box, CircularProgress, Typography, Container } from '@mui/material';
import { Eco as EcoIcon } from '@mui/icons-material';

const LoadingScreen: React.FC = () => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        backgroundColor: 'background.default',
      }}
    >
      <Container maxWidth="sm" sx={{ textAlign: 'center' }}>
        <EcoIcon 
          sx={{ 
            fontSize: 80, 
            color: 'primary.main', 
            mb: 3,
            animation: 'pulse 2s infinite'
          }} 
        />
        
        <Typography variant="h3" component="h1" gutterBottom color="primary">
          AnantaNetra
        </Typography>
        
        <Typography variant="h6" color="text.secondary" gutterBottom>
          AI-Powered Environmental Monitoring
        </Typography>
        
        <Box sx={{ my: 4 }}>
          <CircularProgress size={60} thickness={4} color="primary" />
        </Box>
        
        <Typography variant="body1" color="text.secondary">
          Initializing real-time air quality monitoring system...
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          Protecting millions of lives with AI-powered environmental insights
        </Typography>
      </Container>
      
      <style jsx>{`
        @keyframes pulse {
          0% {
            transform: scale(1);
            opacity: 1;
          }
          50% {
            transform: scale(1.1);
            opacity: 0.7;
          }
          100% {
            transform: scale(1);
            opacity: 1;
          }
        }
      `}</style>
    </Box>
  );
};

export default LoadingScreen;
