import React, { createContext, useContext } from 'react';
import { AppContextType } from '../types';

// Create the context
const AppContext = createContext<AppContextType | undefined>(undefined);

// Custom hook to use the context
export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppContext.Provider');
  }
  return context;
};

export default AppContext;
