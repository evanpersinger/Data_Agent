import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 3006 and 8017 are registered to this project in ~/dotfiles/.claude/PORTS.md.
// strictPort makes a busy port a boot failure; without it Vite silently moves to
// the next free number and the registry is wrong while everything looks fine.
// The proxy keeps the browser same-origin, so the backend needs no CORS config.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3006,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://localhost:8017",
        changeOrigin: true,
      },
    },
  },
});
