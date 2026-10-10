/* The workbench shell: which screen is on, the address bar, the navigation
 * (a drawer on phones), each screen's Inspector (a column on wide screens, a
 * drawer below 1100 px), the page-header state, Help dialogs, theme, language
 * and toasts.  Screens keep their own logic (app.js, pipeline.js, annotate.js)
 * and talk to the shell through window.PureSoundShell. */
(() => {
  "use strict";

  const R = window.PureSoundRoutes;
  const I = window.PureSoundI18n;
  const t = (key, vars) => (I ? I.t(key, vars) : key);
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const store = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage may be disabled */ } },
  };

  const SCREEN_KEY = "puresound.screen";
  const INSPECTOR_KEY = "puresound.inspector.";
  const THEME_KEY = "puresound.theme";
  const THEMES = ["auto", "light", "dark"];
  const DEFAULT_SCREEN = "playground";
  const TITLES = {
    playground: "Playground", verify: "Verify", compare: "Compare", annotate: "Annotate",
    world: "Acoustic world", pipeline: "Pipeline", models: "Models", history: "History",
  };
  // Must match the drawer breakpoints in css/shell.css.
  const narrowInspector = window.matchMedia("(max-width: 1099px)");
  const narrowNav = window.matchMedia("(max-width: 759px)");
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

  const state = { screen: null, drawer: null, returnFocus: null, toastTimer: null, stateTimers: {} };
  const listeners = {};

  // ----------------------------------------------------------------- screens

  function panel(screen) { return $(`[data-screen-panel="${screen}"]`); }

  function show(screen, { updateHash = true, replace = false } = {}) {
    if (!R.SCREENS.includes(screen) || !panel(screen)) screen = DEFAULT_SCREEN;
    const changed = screen !== state.screen;
    closeInspector({ restoreFocus: false });
    closeNav();
    state.screen = screen;
    $$("[data-screen-panel]").forEach((item) => item.classList.toggle("is-visible", item.dataset.screenPanel === screen));
    $$(".nav-link").forEach((link) => {
      const active = link.dataset.screen === screen;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    $("#mobile-title").textContent = t(TITLES[screen]);
    document.title = `${t(TITLES[screen])} · PureSound`;
    store.set(SCREEN_KEY, screen);
    syncInspector(screen);
    const hash = R.hashFor(screen);
    if (updateHash && location.hash !== hash) history[replace ? "replaceState" : "pushState"](null, "", hash);
    if (changed) window.scrollTo({ top: 0 });
    measureHeader();
    if (changed) (listeners[screen] || []).forEach((listener) => listener());
    return screen;
  }

  function current() { return state.screen; }

  function onShow(screen, listener) {
    (listeners[screen] = listeners[screen] || []).push(listener);
    if (state.screen === screen) listener();
  }

  function route() {
    const resolved = R.resolveRoute(location.hash);
    if (resolved) { show(resolved.screen, { updateHash: resolved.redirected, replace: true }); return; }
    const stored = store.get(SCREEN_KEY);
    show(R.SCREENS.includes(stored) ? stored : DEFAULT_SCREEN, { replace: true });
  }

  // --------------------------------------------------------------- inspector

  function inspector(screen) { return $(`#inspector-${screen}`); }
  function toggleButton(screen) { return $(`[data-inspector-toggle="${screen}"]`); }

  /* Wide screens: a column, collapsed or not as last left.  Narrow: closed. */
  function syncInspector(screen = state.screen) {
    const aside = inspector(screen);
    const workbench = aside?.closest(".workbench");
    if (!aside || !workbench) return;
    const collapsed = !narrowInspector.matches && store.get(INSPECTOR_KEY + screen) === "closed";
    workbench.classList.toggle("is-collapsed", collapsed);
    const open = narrowInspector.matches ? state.drawer === screen : !collapsed;
    toggleButton(screen)?.setAttribute("aria-expanded", open ? "true" : "false");
    aside.setAttribute("aria-hidden", open ? "false" : "true");
    aside.inert = !open;
  }

  function openInspector(screen = state.screen, { focus = true } = {}) {
    const aside = inspector(screen);
    if (!aside) return;
    if (narrowInspector.matches) {
      state.returnFocus = document.activeElement;
      state.drawer = screen;
      // The drawer opens under the page header, so the primary action stays
      // in reach; everything else behind it is out of the keyboard's way too.
      const top = `${Math.max(0, Math.round(panel(screen)?.querySelector(".page-header")?.getBoundingClientRect().bottom || 0))}px`;
      aside.style.setProperty("--drawer-top", top);
      aside.classList.add("is-open");
      document.body.classList.add("has-drawer");
      const backdrop = $("#inspector-backdrop");
      backdrop.style.setProperty("--drawer-top", top);
      backdrop.hidden = false;
      requestAnimationFrame(() => backdrop.classList.add("is-open"));
      setBehindInert(screen, true);
    } else {
      store.set(INSPECTOR_KEY + screen, "open");
    }
    syncInspector(screen);
    if (focus) requestAnimationFrame(() => (aside.querySelector("[data-inspector-focus]") || aside.querySelector("select, input, button, textarea"))?.focus({ preventScroll: !narrowInspector.matches }));
  }

  function closeInspector({ restoreFocus = true } = {}) {
    const screen = state.drawer;
    if (!screen) return;
    state.drawer = null;
    setBehindInert(screen, false);
    inspector(screen)?.classList.remove("is-open");
    document.body.classList.remove("has-drawer");
    const backdrop = $("#inspector-backdrop");
    backdrop.classList.remove("is-open");
    window.setTimeout(() => { if (!state.drawer) backdrop.hidden = true; }, 220);
    syncInspector(screen);
    if (restoreFocus) (state.returnFocus?.isConnected ? state.returnFocus : toggleButton(screen))?.focus();
    state.returnFocus = null;
  }

  /* What lies behind the open drawer, apart from the page header. */
  function setBehindInert(screen, on) {
    if (!on) { (state.inert || []).forEach((element) => { element.inert = false; }); state.inert = []; return; }
    const section = panel(screen);
    const workbench = inspector(screen)?.closest(".workbench");
    state.inert = [$("#sidebar"), $(".mobile-bar"),
      ...[...(section?.children || [])].filter((element) => !element.classList.contains("page-header") && element !== workbench),
      ...[...(workbench?.children || [])].filter((element) => !element.classList.contains("inspector"))].filter(Boolean);
    state.inert.forEach((element) => { element.inert = true; });
  }

  function toggleInspector(screen = state.screen) {
    if (narrowInspector.matches) {
      if (state.drawer === screen) closeInspector();
      else openInspector(screen);
      return;
    }
    const collapsed = inspector(screen)?.closest(".workbench")?.classList.contains("is-collapsed");
    if (collapsed) openInspector(screen, { focus: false });
    else { store.set(INSPECTOR_KEY + screen, "closed"); syncInspector(screen); }
  }

  // -------------------------------------------------------------- navigation

  function openNav() {
    if (!narrowNav.matches) return;
    document.body.classList.add("nav-open");
    $("#nav-toggle").setAttribute("aria-expanded", "true");
    const backdrop = $("#nav-backdrop");
    backdrop.hidden = false;
    requestAnimationFrame(() => backdrop.classList.add("is-open"));
    $(".nav-link.is-active")?.focus();
  }

  function closeNav() {
    if (!document.body.classList.contains("nav-open")) return;
    document.body.classList.remove("nav-open");
    $("#nav-toggle").setAttribute("aria-expanded", "false");
    const backdrop = $("#nav-backdrop");
    backdrop.classList.remove("is-open");
    window.setTimeout(() => { if (!document.body.classList.contains("nav-open")) backdrop.hidden = true; }, 220);
  }

  function setSidebarCollapsed(collapsed) {
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    const button = $("#sidebar-toggle");
    button.setAttribute("aria-expanded", collapsed ? "false" : "true");
    button.setAttribute("aria-label", t(collapsed ? "Expand navigation" : "Collapse navigation"));
    button.textContent = collapsed ? "›" : "‹";
    store.set("puresound.sidebar-collapsed", collapsed ? "1" : "0");
  }

  // ------------------------------------------------------ header and state

  /* The sticky header's height, so the Inspector sticks just under it. */
  function measureHeader() {
    const header = panel(state.screen)?.querySelector(".page-header");
    if (header) document.documentElement.style.setProperty("--page-header-h", `${Math.ceil(header.getBoundingClientRect().height)}px`);
  }

  const STATE_LABELS = { run: "Running", ok: "Done", err: "Failed" };

  /* The work a screen is doing, under its header: idle hides it; run shows a
   * chip, a label and progress (0–1); ok and err say how it ended. */
  function setState(screen, { state: kind = "idle", label = "", progress = null } = {}) {
    const row = $(`[data-state-for="${screen}"]`);
    if (!row) return;
    clearTimeout(state.stateTimers[screen]);
    if (kind === "idle") { row.hidden = true; measureHeader(); return; }
    row.hidden = false;
    row.dataset.state = kind;
    const chip = $("[data-state-chip]", row);
    chip.className = `state is-${kind}`;
    chip.textContent = t(STATE_LABELS[kind] || kind);
    $("[data-state-label]", row).textContent = label;
    const bar = $("[data-state-bar]", row);
    const percent = $("[data-state-percent]", row);
    const known = kind === "run" && Number.isFinite(progress);
    bar.hidden = kind !== "run";
    bar.classList.toggle("is-indeterminate", kind === "run" && !known);
    $("span", bar).style.width = known ? `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%` : "";
    percent.textContent = known ? `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%` : "";
    if (kind === "ok") state.stateTimers[screen] = window.setTimeout(() => { row.hidden = true; measureHeader(); }, 6000);
    measureHeader();
  }

  // ------------------------------------------------------------ help, toast

  function openHelp(screen = state.screen) {
    const dialog = $(`#help-${screen}`);
    if (!dialog) return;
    closeNav();
    dialog.showModal?.();
  }

  /* A popover sits in the top layer, so a toast is seen over an open dialog. */
  function toast(message, isError = false) {
    const element = $("#toast");
    element.textContent = message;
    element.classList.toggle("is-error", isError);
    element.setAttribute("role", isError ? "alert" : "status");
    if (element.showPopover) { try { if (element.matches(":popover-open")) element.hidePopover(); element.showPopover(); } catch { /* no popover support */ } }
    element.classList.add("is-visible");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => {
      element.classList.remove("is-visible");
      state.toastTimer = setTimeout(() => { try { element.hidePopover?.(); } catch { /* already hidden */ } }, 250);
    }, isError ? 7000 : 4200);
  }

  function overlayOpen() {
    return Boolean(state.drawer || document.body.classList.contains("nav-open") || $("dialog[open]"));
  }

  // ------------------------------------------------------- theme, language

  function themeChoice() {
    const stored = store.get(THEME_KEY);
    return THEMES.includes(stored) ? stored : "auto";
  }

  function applyTheme(choice = themeChoice()) {
    const dark = choice === "dark" || (choice === "auto" && systemDark.matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    $$("[data-theme-choice]").forEach((button) => button.setAttribute("aria-pressed", button.dataset.themeChoice === choice ? "true" : "false"));
    $('meta[name="theme-color"]')?.setAttribute("content", dark ? "#05061a" : "#010120");
  }

  function setTheme(choice) {
    if (!THEMES.includes(choice)) return;
    store.set(THEME_KEY, choice);
    applyTheme(choice);
  }

  function syncLanguage() {
    const lang = I?.lang || "en";
    $$("[data-lang]").forEach((button) => button.setAttribute("aria-pressed", button.dataset.lang === lang ? "true" : "false"));
    if (state.screen) {
      $("#mobile-title").textContent = t(TITLES[state.screen]);
      document.title = `${t(TITLES[state.screen])} · PureSound`;
    }
    setSidebarCollapsed(document.body.classList.contains("sidebar-collapsed"));
    $$("[data-state-for]").forEach((row) => {
      const chip = $("[data-state-chip]", row);
      if (row.dataset.state && chip) chip.textContent = t(STATE_LABELS[row.dataset.state] || row.dataset.state);
    });
    measureHeader();
  }

  // ------------------------------------------------------------------ wiring

  function wire() {
    // Back and forward between screens change the hash, so hashchange is enough.
    window.addEventListener("hashchange", route);
    $$("[data-inspector-toggle]").forEach((button) => button.addEventListener("click", () => toggleInspector(button.dataset.inspectorToggle)));
    $$("[data-inspector-close]").forEach((button) => button.addEventListener("click", () => closeInspector()));
    $("#inspector-backdrop").addEventListener("click", () => closeInspector());
    // Empty states that point at the settings open them (delegated: some are re-rendered).
    document.addEventListener("click", (event) => {
      const opener = event.target.closest?.("[data-open-inspector]");
      if (opener) openInspector(opener.dataset.openInspector);
    });
    $$("[data-help]").forEach((button) => button.addEventListener("click", () => openHelp(button.dataset.help)));
    $$(".help-dialog [data-dialog-close]").forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));
    $$(".help-dialog").forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); }));
    $("#nav-toggle").addEventListener("click", () => (document.body.classList.contains("nav-open") ? closeNav() : openNav()));
    $("#nav-backdrop").addEventListener("click", closeNav);
    $$(".nav-link").forEach((link) => link.addEventListener("click", closeNav));
    $("#sidebar-toggle").addEventListener("click", () => setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed")));
    $$("[data-theme-choice]").forEach((button) => button.addEventListener("click", () => setTheme(button.dataset.themeChoice)));
    $$("[data-lang]").forEach((button) => button.addEventListener("click", () => I?.setLang(button.dataset.lang)));
    window.addEventListener("puresound:lang", syncLanguage);
    systemDark.addEventListener("change", () => applyTheme());
    narrowInspector.addEventListener("change", () => { closeInspector({ restoreFocus: false }); syncInspector(); });
    narrowNav.addEventListener("change", closeNav);
    window.addEventListener("resize", measureHeader);
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || $("dialog[open]")) return;
      if (state.drawer) { event.preventDefault(); event.stopImmediatePropagation(); closeInspector(); return; }
      if (document.body.classList.contains("nav-open")) { event.preventDefault(); event.stopImmediatePropagation(); closeNav(); $("#nav-toggle").focus(); }
    }, true);
    setSidebarCollapsed(store.get("puresound.sidebar-collapsed") === "1");
    applyTheme();
    syncLanguage();
    route();
  }

  window.PureSoundShell = {
    show,
    current,
    setState,
    openInspector,
    closeInspector,
    toggleInspector,
    openHelp,
    toast,
    onShow,
    overlayOpen,
  };

  wire();
})();
