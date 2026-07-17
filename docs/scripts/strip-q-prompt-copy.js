// When a user clicks a code copy button, copy the code but remove
// leading REPL prompts (q), >>>, and Python continuation ...).
//
// Add this file to mkdocs extra_javascript as javascript/strip-repl-prompts.js
// and ensure it loads after the Material copy button script.

(function () {

    // Robust selectors covering Material for MkDocs variations
    // (theme versions use slightly different class names)
    const COPY_BUTTON_SELECTORS = [
      '.md-code .md-clipboard',     // older / generic
      'button.md-code__copy',       // theme internal
      'button.md-clipboard',        // fallback
      '.md-code__copy-button'       // some custom builds
    ].join(',');
  
    // Regex to remove leading REPL prompts (allow optional leading whitespace)
    // Matches:
    //   q)
    //   >>>
    //   ...
    const REPL_PROMPT_RE = /^\s*(q\)|>>>|\.\.\.)\s?/;
  
    // Helper: find nearest ancestor matching a selector
    function closest(el, selector) {
      while (el) {
        if (el.matches && el.matches(selector)) return el;
        el = el.parentElement;
      }
      return null;
    }
  
    // Clean the code text by removing REPL prompts at the start of each line
    function stripReplPrompts(text) {
      return text
        .split('\n')
        .map(line => line.replace(REPL_PROMPT_RE, ''))
        .join('\n');
    }
  
    // Retrieve visible code text from a code block container
    function getCodeTextFromBlock(container) {
      if (!container) return '';
  
      // Typical structure:
      // <div class="md-code">
      //   <pre><code>...</code></pre>
      // </div>
      const codeElem =
        container.querySelector('pre > code') ||
        container.querySelector('code');
  
      if (!codeElem) return '';
  
      // Use textContent to preserve visible text only (no markup)
      return codeElem.textContent || '';
    }
  
    // Clipboard write with modern API and legacy fallback
    function writeToClipboard(text) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text);
      }
  
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
  
      try {
        document.execCommand('copy');
        document.body.removeChild(textarea);
        return Promise.resolve();
      } catch (err) {
        document.body.removeChild(textarea);
        return Promise.reject(err);
      }
    }
  
    // Attach copy interception to all copy buttons
    function wireCopyButtons() {
      document.querySelectorAll(COPY_BUTTON_SELECTORS).forEach(btn => {
  
        // Prevent duplicate binding if MutationObserver re-runs
        if (btn.__replPromptBound) return;
        btn.__replPromptBound = true;
  
        btn.addEventListener('click', function (ev) {
          try {
            // Locate the containing code block
            const codeContainer =
              closest(btn, '.md-code') ||
              closest(btn, 'pre') ||
              closest(btn, 'code') ||
              btn.parentElement;
  
            const raw = getCodeTextFromBlock(codeContainer);
  
            if (!raw) {
              // If no code found, allow default behavior
              return;
            }
  
            // Remove REPL prompts before copying
            const cleaned = stripReplPrompts(raw);
  
            // Override default copy behavior
            ev.preventDefault();
            ev.stopPropagation();
  
            writeToClipboard(cleaned).then(() => {
              // Optional: emit event for future UI feedback hooks
              btn.dispatchEvent(
                new CustomEvent('repl-copy-success', { bubbles: true })
              );
            }).catch(() => {
              // On failure, allow theme fallback behavior
            });
  
          } catch (e) {
            // Fail silently and allow default copy behavior
            return;
          }
  
        }, { capture: false });
  
      });
    }
  
    // Initial binding once DOM is ready
    document.addEventListener('DOMContentLoaded', () => {
      wireCopyButtons();
  
      // Re-bind if new code blocks are dynamically added
      const observer = new MutationObserver(() => {
        wireCopyButtons();
      });
  
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    });
  
  })();
  