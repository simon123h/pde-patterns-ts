import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

declare const process: { env: Record<string, string | undefined> } // set by the Pages workflow

export default defineConfig({
  // the Pages workflow serves the site below /<repository>/
  base: process.env.BASE_PATH ?? '/',
  plugins: [vue()],
  build: {
    rollupOptions: {
      // the self-test pages ship with the app so that anyone can run them on their own GPU
      input: ['index.html', 'fft-test.html', 'render-test.html', 'solver-test.html'],
    },
  },
  test: { include: ['tests/**/*.test.ts'] },
})
