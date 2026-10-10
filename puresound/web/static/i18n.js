/* English / Traditional Chinese.  English strings are the keys: markup marks
 * text with data-i18n (and attributes with data-i18n-attr="title,placeholder"),
 * scripts call t("…", {vars}).  A string with no Chinese entry shows its
 * English, never a blank or a key.  Loaded after i18n-zh.js (the dictionary);
 * Node loads it too (test/web/i18n.test.cjs). */
(function (root, factory) {
  const api = factory(root);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PureSoundI18n = api;
})(typeof self !== "undefined" ? self : this, function (root) {
  "use strict";

  const LANGS = ["en", "zh-TW"];
  const STORAGE_KEY = "puresound.lang";
  const ATTR_SOURCE = "data-i18n-src-";
  let dictionary = (root && root.PureSoundI18nZH) || {};
  const listeners = new Set();
  const api = { lang: "en", t, setLang, apply, onChange, normalize, setDictionary, LANGS };

  function normalize(tag) {
    return String(tag || "").toLowerCase().startsWith("zh") ? "zh-TW" : "en";
  }

  function fill(text, vars) {
    if (!vars) return text;
    return text.replace(/\{(\w+)\}/g, (match, name) => (vars[name] === undefined || vars[name] === null ? match : String(vars[name])));
  }

  function t(key, vars) {
    const entry = api.lang === "zh-TW" ? dictionary[key] : null;
    return fill(typeof entry === "string" && entry.trim() ? entry : String(key), vars);
  }

  const clean = (text) => String(text || "").replace(/\s+/g, " ").trim();

  /* Whether `text` is `key` in some language: a marked element whose text a
   * script has since replaced (a status note, a file name) is left alone. */
  function shows(text, key) {
    const shown = clean(text);
    return !shown || shown === key || shown === clean(dictionary[key]);
  }

  /* Text and attributes marked in the markup, in the current language.  The
   * English is kept on the element the first time, so it stays the key. */
  function apply(scope) {
    const base = scope || (typeof document !== "undefined" ? document : null);
    if (!base) return;
    for (const element of base.querySelectorAll("[data-i18n]")) {
      let key = element.getAttribute("data-i18n");
      if (!key) {
        key = clean(element.textContent);
        element.setAttribute("data-i18n", key);
      }
      if (key && shows(element.textContent, key)) element.textContent = t(key);
    }
    for (const element of base.querySelectorAll("[data-i18n-attr]")) {
      for (const name of element.getAttribute("data-i18n-attr").split(",").map((item) => item.trim()).filter(Boolean)) {
        let key = element.getAttribute(ATTR_SOURCE + name);
        if (key === null) {
          key = element.getAttribute(name);
          if (key === null) continue;
          element.setAttribute(ATTR_SOURCE + name, key);
        }
        if (shows(element.getAttribute(name), key)) element.setAttribute(name, t(key));
      }
    }
  }

  function setLang(next, { persist = true } = {}) {
    if (!LANGS.includes(next)) return api.lang;
    if (next === api.lang) return api.lang;
    api.lang = next;
    if (persist) {
      try { root.localStorage.setItem(STORAGE_KEY, next); } catch { /* storage may be disabled */ }
    }
    if (typeof document !== "undefined") {
      document.documentElement.lang = next === "zh-TW" ? "zh-Hant-TW" : "en";
      apply(document);
      root.dispatchEvent(new CustomEvent("puresound:lang", { detail: { lang: next } }));
    }
    for (const listener of [...listeners]) listener(next);
    return next;
  }

  function onChange(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function setDictionary(entries) {
    dictionary = entries || {};
  }

  /* The stored choice, else the browser's language. */
  function initial() {
    try {
      const stored = root.localStorage.getItem(STORAGE_KEY);
      if (LANGS.includes(stored)) return stored;
    } catch { /* storage may be disabled */ }
    const nav = root && root.navigator;
    return normalize((nav && (nav.languages && nav.languages[0])) || (nav && nav.language) || "en");
  }

  if (typeof document !== "undefined") {
    api.lang = initial();
    document.documentElement.lang = api.lang === "zh-TW" ? "zh-Hant-TW" : "en";
    apply(document);
  }
  return api;
});
