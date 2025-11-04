import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Tabs,
  Tab,
  Container,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Map as MapIcon,
  Search as SearchIcon,
  HealthAndSafety as HealthIcon,
} from '@mui/icons-material';

const Navigation: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const currentTab = location.pathname;

  const handleTabChange = (_: React.SyntheticEvent, newValue: string) => {
    navigate(newValue);
  };

  return (
    <Box sx={{ 
      borderBottom: 1, 
      borderColor: 'divider',
      backgroundColor: 'background.paper',
      position: 'sticky',
      top: 64, // Height of AppBar
      zIndex: 100,
    }}>
      <Container maxWidth="xl">
        <Tabs
          value={currentTab}
          onChange={handleTabChange}
          aria-label="navigation tabs"
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTab-root': {
              minWidth: 120,
              textTransform: 'none',
              fontSize: '0.95rem',
              fontWeight: 500,
            },
          }}
        >
          <Tab
            label="Dashboard"
            value="/dashboard"
            icon={<DashboardIcon />}
            iconPosition="start"
            sx={{ 
              '&.Mui-selected': { 
                color: 'primary.main',
                fontWeight: 600,
              } 
            }}
          />
          <Tab
            label="Map View"
            value="/map"
            icon={<MapIcon />}
            iconPosition="start"
            sx={{ 
              '&.Mui-selected': { 
                color: 'primary.main',
                fontWeight: 600,
              } 
            }}
          />
          <Tab
            label="Search Location"
            value="/search"
            icon={<SearchIcon />}
            iconPosition="start"
            sx={{ 
              '&.Mui-selected': { 
                color: 'primary.main',
                fontWeight: 600,
              } 
            }}
          />
          <Tab
            label="Health Advisory"
            value="/health"
            icon={<HealthIcon />}
            iconPosition="start"
            sx={{ 
              '&.Mui-selected': { 
                color: 'primary.main',
                fontWeight: 600,
              } 
            }}
          />
        </Tabs>
      </Container>
    </Box>
  );
};

export default Navigation;
