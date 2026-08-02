import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 5185 and 8017 are free per ~/dotfiles/.claude/PORTS.md.
// The proxy keeps the browser same-origin, so the backend needs no CORS config.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5185,
    proxy: {
      "/api": {
        target: "http://localhost:8017",
        changeOrigin: true,
      },
    },
  },
});
