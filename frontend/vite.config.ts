import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/system": "http://127.0.0.1:8000",
      "/profiles": "http://127.0.0.1:8000",
      "/datasets": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
    },
  },
});
