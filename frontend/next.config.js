const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  // stray lockfiles above this project must not hijack workspace-root inference
  outputFileTracingRoot: __dirname,
  webpack: (config) => {
    config.resolve.alias["@" ] = path.resolve(__dirname);
    return config;
  },
  async rewrites() {
    const backend = process.env.BACKEND_URL || "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

module.exports = nextConfig;
