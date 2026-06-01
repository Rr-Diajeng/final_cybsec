import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // forward /api calls to FastAPI so you don't need CORS in dev
      "/api": {
        target: "http://localhost:8000",
        rewrite: path => path.replace(/^\/api/, ""),
      },
    },
  },
});
