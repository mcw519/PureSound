/* Screens and their addresses (#/playground).  Addresses from before the
 * redesign still land: a bookmark or a link in an old report opens the screen
 * that replaced it.  Pure, so Node tests it (test/web/routes.test.cjs). */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PureSoundRoutes = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const SCREENS = ["playground", "verify", "compare", "annotate", "world", "pipeline", "models", "history"];
  const REDIRECTS = {
    zoo: "models",
    measurements: "compare",
    "playground/voice": "playground",
    "playground/sv": "verify",
  };

  /* {screen, redirected} for a location hash; null when it names no screen,
   * so the caller falls back to the stored or default one. */
  function resolveRoute(hash) {
    const raw = String(hash || "");
    const path = raw.replace(/^#\/?/, "");
    const trimmed = path.replace(/\/+$/, "");
    if (!trimmed) return null;
    const canonical = raw === `#/${trimmed}`;
    if (SCREENS.includes(trimmed)) return { screen: trimmed, redirected: !canonical };
    if (REDIRECTS[trimmed]) return { screen: REDIRECTS[trimmed], redirected: true };
    return null;
  }

  function hashFor(screen) {
    return `#/${screen}`;
  }

  return { SCREENS, REDIRECTS, resolveRoute, hashFor };
});
