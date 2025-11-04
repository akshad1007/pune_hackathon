/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WEATHER_API_KEY: string;
  readonly VITE_OPENWEATHER_API_KEY: string;
  readonly VITE_OPENCAGE_API_KEY: string;
  readonly VITE_GEMINI_API_KEY: string;
  readonly MODE: string;
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
