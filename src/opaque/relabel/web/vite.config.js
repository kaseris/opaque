import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// base './' so the built bundle loads regardless of the path FastAPI mounts it at.
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    // In `npm run dev`, proxy API calls to the FastAPI backend (`opaque relabel`).
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
