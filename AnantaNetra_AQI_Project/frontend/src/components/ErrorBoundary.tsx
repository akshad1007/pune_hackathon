import React, { Component, ReactNode } from 'react';
import { Box, Typography, Button, Container, Alert } from '@mui/material';
import { ErrorOutline as ErrorIcon, Refresh as RefreshIcon } from '@mui/icons-material';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: string | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo: errorInfo.componentStack,
    });
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            backgroundColor: 'background.default',
            p: 3,
          }}
        >
          <Container maxWidth="md">
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <ErrorIcon 
                sx={{ 
                  fontSize: 80, 
                  color: 'error.main', 
                  mb: 2
                }} 
              />
              
              <Typography variant="h4" component="h1" gutterBottom color="error">
                Oops! Something went wrong
              </Typography>
              
              <Typography variant="h6" color="text.secondary" gutterBottom>
                AnantaNetra encountered an unexpected error
              </Typography>
            </Box>

            <Alert severity="error" sx={{ mb: 3 }}>
              <Typography variant="body2">
                <strong>Error:</strong> {this.state.error?.message || 'Unknown error occurred'}
              </Typography>
            </Alert>

            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <Alert severity="warning" sx={{ mb: 3 }}>
                <Typography variant="body2" component="pre" sx={{ 
                  fontFamily: 'monospace', 
                  fontSize: '0.75rem',
                  maxHeight: 200,
                  overflow: 'auto'
                }}>
                  {this.state.errorInfo}
                </Typography>
              </Alert>
            )}

            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<RefreshIcon />}
                onClick={this.handleReload}
                size="large"
              >
                Reload Application
              </Button>
              
              <Button
                variant="outlined"
                color="primary"
                onClick={this.handleReset}
                size="large"
              >
                Try Again
              </Button>
            </Box>

            <Box sx={{ mt: 4, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                If this problem persists, please try:
              </Typography>
              <Typography variant="body2" color="text.secondary" component="ul" sx={{ mt: 1, textAlign: 'left', maxWidth: 400, mx: 'auto' }}>
                <li>Refreshing the page</li>
                <li>Clearing your browser cache</li>
                <li>Checking your internet connection</li>
                <li>Verifying the backend service is running</li>
              </Typography>
            </Box>

            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                AnantaNetra v1.0.0 - AI Environmental Monitoring System
              </Typography>
            </Box>
          </Container>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
