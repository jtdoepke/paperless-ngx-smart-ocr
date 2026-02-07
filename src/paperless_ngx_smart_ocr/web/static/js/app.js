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

/**
 * Modal management for preview and apply flows.
 *
 * Provides window.closeModal() and handles escape key, backdrop click,
 * body scroll lock, and markdown raw/rendered toggle.
 */
(function () {
  "use strict";

  /** Close the modal by clearing #modal-container. */
  window.closeModal = function () {
    var container = document.getElementById("modal-container");
    if (container) container.innerHTML = "";
    document.body.classList.remove("overflow-hidden");
  };

  // Escape key closes modal
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      var container = document.getElementById("modal-container");
      if (container && container.innerHTML.trim()) {
        window.closeModal();
      }
    }
  });

  // Lock body scroll when modal content appears
  document.addEventListener("htmx:afterSwap", function (e) {
    if (e.detail.target && e.detail.target.id === "modal-container") {
      if (e.detail.target.innerHTML.trim()) {
        document.body.classList.add("overflow-hidden");
      }
    }
  });

  // Markdown raw/rendered toggle via delegation
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-md-toggle]");
    if (!btn) return;

    var mode = btn.dataset.mdToggle;
    var rendered = document.getElementById("md-rendered");
    var raw = document.getElementById("md-raw");
    if (!rendered || !raw) return;

    if (mode === "raw") {
      rendered.classList.add("hidden");
      raw.classList.remove("hidden");
    } else {
      rendered.classList.remove("hidden");
      raw.classList.add("hidden");
    }

    // Update toggle button styles
    var buttons = btn.parentElement.querySelectorAll("[data-md-toggle]");
    buttons.forEach(function (b) {
      if (b.dataset.mdToggle === mode) {
        b.className =
          "px-2 py-0.5 text-xs rounded bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400";
      } else {
        b.className =
          "px-2 py-0.5 text-xs rounded text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700";
      }
    });
  });
})();

/**
 * Bulk document selection and processing.
 *
 * Manages checkbox state across htmx table swaps and provides
 * a fixed action bar for bulk processing selected documents.
 */
(function () {
  "use strict";

  // -- Selection state (persists across htmx swaps) --
  var selectedIds = new Set();
  var selectAllMatchingMode = false;
  var allMatchingCount = 0;
  var allMatchingFilterQs = "";

  // -- UI helpers --

  function updateUI() {
    var bar = document.getElementById("bulk-action-bar");
    var text = document.getElementById("bulk-selection-text");
    if (!bar || !text) return;

    var count = selectAllMatchingMode ? allMatchingCount : selectedIds.size;

    if (count > 0) {
      bar.classList.remove("hidden");
      var suffix = selectAllMatchingMode ? " (all matching)" : "";
      text.textContent =
        count + " document" + (count !== 1 ? "s" : "") + " selected" + suffix;
    } else {
      bar.classList.add("hidden");
    }
    syncHeaderCheckbox();
  }

  function syncHeaderCheckbox() {
    var headerCb = document.getElementById("select-page");
    if (!headerCb) return;
    var rowCbs = document.querySelectorAll(".doc-select");
    var allChecked =
      rowCbs.length > 0 &&
      Array.from(rowCbs).every(function (cb) {
        return cb.checked;
      });
    var someChecked = Array.from(rowCbs).some(function (cb) {
      return cb.checked;
    });
    headerCb.checked = allChecked;
    headerCb.indeterminate = someChecked && !allChecked;
  }

  function recheckBoxes() {
    var rowCbs = document.querySelectorAll(".doc-select");
    rowCbs.forEach(function (cb) {
      var id = parseInt(cb.dataset.documentId, 10);
      cb.checked = selectAllMatchingMode || selectedIds.has(id);
    });
    syncHeaderCheckbox();
  }

  function clearSelection() {
    selectedIds.clear();
    selectAllMatchingMode = false;
    allMatchingCount = 0;
    allMatchingFilterQs = "";
    document.querySelectorAll(".doc-select").forEach(function (cb) {
      cb.checked = false;
    });
    updateUI();
  }

  function readTableData() {
    var table = document.querySelector("#document-table table");
    if (!table) return {};
    return {
      totalCount: parseInt(table.dataset.totalCount, 10) || 0,
      filterQs: table.dataset.filterQs || "",
    };
  }

  // -- Event handlers (all use delegation) --

  // Row checkboxes: delegate on #document-table
  document.addEventListener("change", function (e) {
    if (!e.target.classList.contains("doc-select")) return;
    var id = parseInt(e.target.dataset.documentId, 10);
    if (e.target.checked) {
      selectedIds.add(id);
    } else {
      selectedIds.delete(id);
      selectAllMatchingMode = false;
    }
    updateUI();
  });

  // Header "select page" checkbox
  document.addEventListener("change", function (e) {
    if (e.target.id !== "select-page") return;
    document.querySelectorAll(".doc-select").forEach(function (cb) {
      cb.checked = e.target.checked;
      var id = parseInt(cb.dataset.documentId, 10);
      if (e.target.checked) {
        selectedIds.add(id);
      } else {
        selectedIds.delete(id);
      }
    });
    if (!e.target.checked) {
      selectAllMatchingMode = false;
    }
    updateUI();
  });

  // All click-based actions via single delegated listener
  document.addEventListener("click", function (e) {
    var target = e.target.closest("[id]");
    if (!target) return;

    // Dropdown toggle
    if (target.id === "select-dropdown-toggle") {
      var dd = document.getElementById("select-dropdown");
      if (dd) dd.classList.toggle("hidden");
      return;
    }

    // Dropdown: select this page
    if (target.id === "select-all-page") {
      document.querySelectorAll(".doc-select").forEach(function (cb) {
        cb.checked = true;
        selectedIds.add(parseInt(cb.dataset.documentId, 10));
      });
      selectAllMatchingMode = false;
      updateUI();
      document.getElementById("select-dropdown").classList.add("hidden");
      return;
    }

    // Dropdown: select all matching
    if (target.id === "select-all-matching") {
      var data = readTableData();
      allMatchingCount = data.totalCount;
      allMatchingFilterQs = data.filterQs;
      selectAllMatchingMode = true;
      document.querySelectorAll(".doc-select").forEach(function (cb) {
        cb.checked = true;
      });
      updateUI();
      document.getElementById("select-dropdown").classList.add("hidden");
      return;
    }

    // Dropdown: clear selection
    if (target.id === "deselect-all") {
      clearSelection();
      document.getElementById("select-dropdown").classList.add("hidden");
      return;
    }

    // Bulk action bar: clear
    if (target.id === "bulk-clear-btn") {
      clearSelection();
      return;
    }

    // Bulk action bar: process (opens preview review modal)
    if (target.id === "bulk-process-btn") {
      var values = {};

      if (selectAllMatchingMode) {
        values.filter_query = allMatchingFilterQs;
      } else {
        values.document_ids = Array.from(selectedIds).join(",");
      }

      htmx.ajax("POST", "/documents/bulk-preview", {
        target: "#modal-container",
        swap: "innerHTML",
        values: values,
      });
      clearSelection();
      return;
    }
  });

  // Close dropdown on outside click
  document.addEventListener("click", function (e) {
    var dd = document.getElementById("select-dropdown");
    if (!dd || dd.classList.contains("hidden")) return;
    var toggle = document.getElementById("select-dropdown-toggle");
    if (
      !dd.contains(e.target) &&
      (!toggle || !toggle.contains(e.target))
    ) {
      dd.classList.add("hidden");
    }
  });

  // Re-check boxes after htmx swaps the document table
  document.addEventListener("htmx:afterSwap", function (e) {
    if (e.detail.target && e.detail.target.id === "document-table") {
      recheckBoxes();
      updateUI();
    }
  });

  // Clear selection when filter form is submitted (new filter = new universe)
  document.addEventListener("htmx:configRequest", function (e) {
    if (e.detail.elt && e.detail.elt.closest("#filter-form")) {
      clearSelection();
    }
  });
})();
