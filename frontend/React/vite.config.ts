import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 8080,
    proxy: {
      '/solve': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        timeout: 0,
        proxyTimeout: 0,
        // http-proxy rewrites Connection: keep-alive → close, which causes
        // Firefox to terminate the SSE stream immediately (Error in input stream).
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['connection'] = 'keep-alive';
          });
        },
      },
      '/history': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        timeout: 0,
        proxyTimeout: 0,
      },
      '/stop': {
        target: 'http://localhost:8002',
        changeOrigin: true,
      },
    },
  },
})
