document.addEventListener("DOMContentLoaded", function () {
  var script = document.createElement("script");
  script.src = "https://widget.kapa.ai/kapa-widget.bundle.js";

  // ── Core / identity ───────────────────────────────────────────────
  script.setAttribute("data-website-id", "90e95ec3-2524-43ab-93da-3e94fdd01d59");
  script.setAttribute("data-project-name", "KX");
  script.setAttribute("data-modal-title", "Ask AI");

  // ── Branding: logos ───────────────────────────────────────────────
  // Launcher / button logo:
  script.setAttribute("data-project-logo", "https://kx.com/files/kx-logo-dark.png");
  script.setAttribute("data-project-logo-dark", "https://kx.com/files/kx-logo-white.png");
  // Modal (answer window) logo — the widget reads `-src` (dark handled via the
  // generic `-dark` suffix). Bare `data-modal-logo` is not read by the widget.
  // Pick the initial `-src` based on the page's active MkDocs color scheme:
  // white logo when slate (dark) is active, otherwise the dark logo.
  // NB: read the ACTIVE scheme off <body> — the palette toggle renders hidden
  // inputs each carrying a `data-md-color-scheme` attribute, so a generic
  // `[data-md-color-scheme='slate']` query always matches and misreports light.
  var isSlate = document.body.getAttribute("data-md-color-scheme") === "slate";
  script.setAttribute(
    "data-modal-logo-src",
    isSlate
      ? "https://kx.com/files/kx-logo-white.png"
      : "https://kx.com/files/kx-logo-dark.png"
  );
  //script.setAttribute("data-modal-logo-src-dark", "https://kx.com/files/kx-logo-white.png");

  // ── Branding: accent color (gold in light, blue in dark) ──────────
  // Drives the primary/brand accent (e.g. buttons). Legacy attribute,
  // retained until kapa exposes a dedicated semantic accent token.
  script.setAttribute("data-project-color", "#38a8e0");
  script.setAttribute("data-project-color-dark", "#FFC300");
  script.setAttribute("data-button-text-color", "#060000");

  // Some components (the Deep Thinking toggle when enabled, and the Stop
  // button) derive their color from the *base* `data-project-color` inline and
  // ignore `data-project-color-dark`, so they stay blue in dark mode. Component
  // styles have no `-dark` variant, so we apply gold overrides only when slate
  // is the active scheme at load (pages reload on navigation, so this stays
  // correct without a live observer).
  if (isSlate) {
    script.setAttribute("data-deep-thinking-button-enabled-color", "#FFC300");
    script.setAttribute("data-deep-thinking-button-enabled-background-color", "rgba(255, 195, 0, 0.1)");
    script.setAttribute("data-deep-thinking-button-enabled-border-color", "#FFC300");
    // Submit / stop button (solid-fill icon button) — gold fill in dark.
    script.setAttribute("data-submit-button-background-color", "#FFC300");
  }

  // ── Layout & sizing ───────────────────────────────────────────────
  script.setAttribute("data-view-mode", "sidebar");
  script.setAttribute("data-modal-open-by-default", "false");
  script.setAttribute("data-modal-z-index", "10001");
  script.setAttribute("data-modal-size", "495px"); 
  script.setAttribute("data-scale-factor", "0.8");
  script.setAttribute("data-modal-title-font-size", "18px");

  // ── Theme: color scheme (default light; follow MkDocs slate → dark) ─
  script.setAttribute("data-color-scheme", "light");
  script.setAttribute("data-color-scheme-selector", "[data-md-color-scheme='slate']");

   // ── Theme: semantic tokens — dark ────────────────────────────────
  script.setAttribute("data-surface-color-dark", "#141417");
  script.setAttribute("data-surface-elevated-color-dark", "#1f1f24");
  script.setAttribute("data-surface-hover-color-dark", "#2a2a31");
  script.setAttribute("data-text-color-dark", "#e8e8ec");
  script.setAttribute("data-text-muted-color-dark", "#9ca3af");
  script.setAttribute("data-border-color-dark", "#4a4a52");
  script.setAttribute("data-anchor-color-dark", "#B8860B"); // darkened gold: legible link color on dark  
  script.setAttribute("data-deep-thinking-button-enabled-color-dark", "#B8860B");    
  script.setAttribute("data-deep-thinking-button-hover-color-dark", "#B8860B"); 
  script.setAttribute("data-deep-thinking-button-enabled-hover-color-dark", "#B8860B");    
  script.setAttribute("data-answer-feedback-button-enabled-border-dark", "1px solid #B8860B");  
  script.setAttribute("data-answer-feedback-button-enabled-color-dark", "#B8860B");    
  script.setAttribute("data-answer-feedback-button-enabled-hover-color-dark", "#B8860B");     

  // ── Theme: semantic tokens — light ────────────────────────────────
  script.setAttribute("data-surface-color", "#ffffff");
  script.setAttribute("data-surface-elevated-color", "#f8f9fa");
  script.setAttribute("data-surface-hover-color", "#f1f3f5");
  script.setAttribute("data-text-color", "#111111");
  script.setAttribute("data-text-muted-color", "#666666");
  script.setAttribute("data-border-color", "#dee2e6");
  script.setAttribute("data-anchor-color", "#5ab8f0"); // blue: matches light accent  
  script.setAttribute("data-deep-thinking-button-enabled-color", "#5ab8f0");  
  script.setAttribute("data-deep-thinking-button-hover-color", "#5ab8f0");  
  script.setAttribute("data-deep-thinking-button-enabled-hover-color", "#5ab8f0");    
  script.setAttribute("data-deep-thinking-button-text-hover-color", "#5ab8f0");    
  script.setAttribute("data-answer-feedback-button-enabled-border", "1px solid #5ab8f0"); 
  script.setAttribute("data-answer-feedback-button-enabled-color", "#5ab8f0");      
  script.setAttribute("data-answer-feedback-button-enabled-hover-color", "#5ab8f0");   

  // ── Features: MCP & search ────────────────────────────────────────
  script.setAttribute("data-mcp-enabled", "true");
  script.setAttribute("data-mcp-server-url", "https://kx.mcp.kapa.ai");
  script.setAttribute("data-search-mode-enabled", "true");
  script.setAttribute("data-search-source-ids-order", "c2743780-4e53-447c-a7f9-b12c419907c2");

  // ── Analytics & consent ───────────────────────────────────────────
  script.setAttribute("data-user-analytics-fingerprint-enabled", "false");
  script.setAttribute("data-user-analytics-cookie-enabled", "false");
  script.setAttribute("data-consent-required", "false");
  script.setAttribute("data-consent-screen-title", "Help us improve our AI assistant");
  script.setAttribute("data-consent-screen-disclaimer", "By clicking 'Allow tracking', you consent to anonymous user tracking which helps us improve our service. We don't collect any personally identifiable information.");
  script.setAttribute("data-consent-screen-accept-button-text", "Allow tracking");
  script.setAttribute("data-consent-screen-reject-button-text", "No, thanks");

  // ── Content: disclaimer & example questions ───────────────────────
  script.setAttribute("data-modal-disclaimer", "This is an experimental custom LLM for answering technical questions about KX products. Answers are based **only** on KX documentation and support sources, but may not be fully accurate so please use your best judgement. Don't forget to rate the answers and give feedback!");
  script.setAttribute("data-modal-example-questions", "How do I get started with KDB-X Python?,What's new in KDB-X Python (pykx)?,How do I get started with KDB-X?,How do I learn q?");  

  script.async = true;
  document.head.appendChild(script);
});
