document$.subscribe(function () {
  const THRESHOLD = 25;
  const COLLAPSE_LINES = 10;

  document.querySelectorAll("pre code").forEach(function (code) {
    const pre = code.parentElement;

    if (pre.closest(".code-compare, .code-compare-three")) return;

    const lines = code.textContent.split("\n").filter(function (l, i, arr) {
      return !(i === arr.length - 1 && l === "");
    });
    if (lines.length < THRESHOLD) return;

    const wrapper = document.createElement("div");
    wrapper.className = "code-collapse";
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    const lineHeight = parseFloat(getComputedStyle(code).lineHeight);
    const collapsedHeight = lineHeight * COLLAPSE_LINES;
    pre.style.maxHeight = collapsedHeight + "px";
    pre.style.overflow = "hidden";

    const toggle = document.createElement("button");
    toggle.className = "code-collapse-toggle code-collapse-toggle--bottom";
    toggle.setAttribute("aria-expanded", "false");
    toggle.textContent = "▼  Show all " + lines.length + " lines";
    wrapper.appendChild(toggle);

    function expand() {
      pre.style.maxHeight = "none";
      pre.style.overflow = "";
      toggle.setAttribute("aria-expanded", "true");
      toggle.textContent = "▲  Show fewer lines";
      wrapper.classList.add("code-collapse--expanded");
    }

    function collapse() {
      pre.style.maxHeight = collapsedHeight + "px";
      pre.style.overflow = "hidden";
      toggle.setAttribute("aria-expanded", "false");
      toggle.textContent = "▼  Show all " + lines.length + " lines";
      wrapper.classList.remove("code-collapse--expanded");
    }

    toggle.addEventListener("click", function () {
      pre.style.overflow === "hidden" ? expand() : collapse();
    });

    // Auto-expand if search.highlight injects a <mark> inside a collapsed block.
    // MutationObserver catches marks added after collapse runs (search nav is async).
    if (pre.querySelector("mark")) {
      expand();
    } else {
      var observer = new MutationObserver(function () {
        if (pre.querySelector("mark")) {
          expand();
          observer.disconnect();
        }
      });
      observer.observe(pre, { childList: true, subtree: true });
    }
  });
});
