/**
 * Tailwind CSS configuration for paperless-ngx-smart-ocr.
 *
 * This file is used by the Tailwind standalone CLI for production builds:
 *   tailwindcss -i src/input.css -o src/paperless_ngx_smart_ocr/web/static/css/app.css --minify
 *
 * During development, Tailwind is loaded via CDN with inline config in base.html.
 *
 * @type {import('tailwindcss').Config}
 */
module.exports = {
  darkMode: "class",
  content: [
    "./src/paperless_ngx_smart_ocr/web/templates/**/*.html",
    "./src/paperless_ngx_smart_ocr/web/static/js/**/*.js",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
