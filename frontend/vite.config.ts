import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  envDir: "..",
  server: {
    proxy: {
      "/api": "https://competitions-css-tested-weed.trycloudflare.com",
    },
  },
});
