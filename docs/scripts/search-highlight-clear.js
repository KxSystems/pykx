// Clear search.highlight marks on first click after they appear.
// Marks are injected asynchronously after navigation, so a MutationObserver
// watches for them rather than polling with a fixed timeout.
document$.subscribe(function () {
  var content = document.querySelector(".md-content") || document.body;

  var observer = new MutationObserver(function () {
    if (!content.querySelector("mark")) return;
    observer.disconnect();
    document.addEventListener("click", function clearMarks() {
      content.querySelectorAll("mark").forEach(function (mark) {
        mark.replaceWith(document.createTextNode(mark.textContent));
      });
      document.removeEventListener("click", clearMarks);
    });
  });

  observer.observe(content, { childList: true, subtree: true });
});
