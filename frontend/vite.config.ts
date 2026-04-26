import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // proxies any path beginning with /predict (covers /predict, /predict/hyundai, /predict/engine)
      '/predict': 'http://localhost:8000',
      '/model_info': 'http://localhost:8000',
      '/drift': 'http://localhost:8000',
      '/train': 'http://localhost:8000',
    },
  },
})
