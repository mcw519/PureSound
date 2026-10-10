// The English / Traditional Chinese string layer, run by test_web_js.py with `node --test`.
const test = require("node:test");
const assert = require("node:assert/strict");
const I = require("../../puresound/web/static/i18n.js");

const ZH = { Run: "執行", "Clip {n} of {total}": "第 {n} 段，共 {total} 段", "Open audio": "開啟音檔", "Search models": "搜尋模型" };

function element(attrs = {}, text = "") {
  const store = { ...attrs };
  return {
    textContent: text,
    getAttribute: (name) => (name in store ? store[name] : null),
    setAttribute: (name, value) => { store[name] = String(value); },
    hasAttribute: (name) => name in store,
    attrs: store,
  };
}

function fragment(elements) {
  return {
    querySelectorAll(selector) {
      const name = selector.slice(1, -1);
      return elements.filter((el) => el.hasAttribute(name));
    },
  };
}

test.beforeEach(() => {
  I.setDictionary(ZH);
  I.setLang("en", { persist: false });
});

test("English returns the key itself", () => {
  assert.equal(I.t("Run"), "Run");
  assert.equal(I.t("Nothing translated"), "Nothing translated");
});

test("Chinese returns the entry and falls back to English when one is missing", () => {
  I.setLang("zh-TW", { persist: false });
  assert.equal(I.lang, "zh-TW");
  assert.equal(I.t("Run"), "執行");
  assert.equal(I.t("Not in the dictionary"), "Not in the dictionary");
});

test("an empty entry never blanks a string", () => {
  I.setDictionary({ Run: "" });
  I.setLang("zh-TW", { persist: false });
  assert.equal(I.t("Run"), "Run");
});

test("placeholders fill in both languages and a missing one stays visible", () => {
  assert.equal(I.t("Clip {n} of {total}", { n: 2, total: 5 }), "Clip 2 of 5");
  I.setLang("zh-TW", { persist: false });
  assert.equal(I.t("Clip {n} of {total}", { n: 2, total: 5 }), "第 2 段，共 5 段");
  assert.equal(I.t("Clip {n} of {total}", { n: 2 }), "第 2 段，共 {total} 段");
  assert.equal(I.t("Clip {n} of {total}", { n: 0, total: 0 }), "第 0 段，共 0 段");
});

test("language tags normalise to the two supported languages", () => {
  assert.equal(I.normalize("zh-Hant-TW"), "zh-TW");
  assert.equal(I.normalize("zh-TW"), "zh-TW");
  assert.equal(I.normalize("zh-CN"), "zh-TW");
  assert.equal(I.normalize("zh"), "zh-TW");
  assert.equal(I.normalize("en-US"), "en");
  assert.equal(I.normalize("fr"), "en");
  assert.equal(I.normalize(undefined), "en");
});

test("an unsupported language is refused and the current one kept", () => {
  I.setLang("zh-TW", { persist: false });
  I.setLang("de", { persist: false });
  assert.equal(I.lang, "zh-TW");
});

test("apply keeps the English as the key, so switching back restores it", () => {
  const label = element({ "data-i18n": "" }, "  Open\n   audio ");
  const keyed = element({ "data-i18n": "Run" }, "Run");
  const search = element({ "data-i18n-attr": "placeholder,aria-label", placeholder: "Search models", "aria-label": "Search models" });
  const root = fragment([label, keyed, search]);

  I.setLang("zh-TW", { persist: false });
  I.apply(root);
  assert.equal(label.textContent, "開啟音檔");
  assert.equal(keyed.textContent, "執行");
  assert.equal(search.attrs.placeholder, "搜尋模型");
  assert.equal(search.attrs["aria-label"], "搜尋模型");

  I.setLang("en", { persist: false });
  I.apply(root);
  assert.equal(label.textContent, "Open audio");
  assert.equal(keyed.textContent, "Run");
  assert.equal(search.attrs.placeholder, "Search models");

  // Applying twice in one language is stable.
  I.setLang("zh-TW", { persist: false });
  I.apply(root);
  I.apply(root);
  assert.equal(label.textContent, "開啟音檔");
});

test("apply leaves alone a marked element whose text a script replaced", () => {
  const note = element({ "data-i18n": "Run" }, "Run");
  const title = element({ "data-i18n-attr": "title", title: "Open audio" });
  const root = fragment([note, title]);
  I.setLang("zh-TW", { persist: false });
  I.apply(root);
  note.textContent = "two-talkers.wav is ready.";
  title.setAttribute("title", "Wait for the run to finish");
  I.setLang("en", { persist: false });
  I.apply(root);
  assert.equal(note.textContent, "two-talkers.wav is ready.");
  assert.equal(title.attrs.title, "Wait for the run to finish");
});

test("listeners hear a change once and can unsubscribe", () => {
  const heard = [];
  const off = I.onChange((lang) => heard.push(lang));
  I.setLang("zh-TW", { persist: false });
  I.setLang("zh-TW", { persist: false });
  off();
  I.setLang("en", { persist: false });
  assert.deepEqual(heard, ["zh-TW"]);
});

test("the shipped dictionary has no empty entries", () => {
  const zh = require("../../puresound/web/static/i18n-zh.js");
  const empty = Object.entries(zh).filter(([, value]) => typeof value !== "string" || !value.trim());
  assert.deepEqual(empty, []);
  assert.ok(Object.keys(zh).length > 10);
});
