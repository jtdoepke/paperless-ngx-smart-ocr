/**
 * paperless-ngx-smart-ocr UI scripts.
 *
 * Handles dark mode toggling and htmx event integration.
 */

(function () {
  "use strict";

  var THEME_KEY = "smart-ocr-theme";

  /**
   * Determine the effective theme based on server config, user preference,
   * and system setting.
   */
  function getEffectiveTheme() {
    var stored = localStorage.getItem(THEME_KEY);
    if (stored === "dark" || stored === "light") {
      return stored;
    }
    var serverMode = document.documentElement.dataset.themeMode;
    if (serverMode === "dark") return "dark";
    if (serverMode === "light") return "light";
    // auto: follow system preference
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  /** Apply the given theme by toggling the `dark` class on <html>. */
  function applyTheme(theme) {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }

  /** Toggle between dark and light theme, persisting the choice. */
  function toggleTheme() {
    var current = document.documentElement.classList.contains("dark")
      ? "dark"
      : "light";
    var next = current === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
    updateToggleIcon(next);
  }

  /** Update the sun/moon icon in the toggle button. */
  function updateToggleIcon(theme) {
    var sunIcon = document.getElementById("theme-icon-sun");
    var moonIcon = document.getElementById("theme-icon-moon");
    if (sunIcon && moonIcon) {
      sunIcon.classList.toggle("hidden", theme !== "dark");
      moonIcon.classList.toggle("hidden", theme === "dark");
    }
  }

  // Apply theme immediately (called from inline script in <head> too)
  applyTheme(getEffectiveTheme());

  // Once DOM is ready, set up toggle button and icons
  document.addEventListener("DOMContentLoaded", function () {
    updateToggleIcon(getEffectiveTheme());

    var btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.addEventListener("click", toggleTheme);
    }
  });

  // Listen for system preference changes (when in auto mode)
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", function (e) {
      if (!localStorage.getItem(THEME_KEY)) {
        var serverMode = document.documentElement.dataset.themeMode;
        if (!serverMode || serverMode === "auto") {
          applyTheme(e.matches ? "dark" : "light");
          updateToggleIcon(e.matches ? "dark" : "light");
        }
      }
    });
})();
