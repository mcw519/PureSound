// Screen routes and the old addresses that still have to land, run by test_web_js.py.
const test = require("node:test");
const assert = require("node:assert/strict");
const R = require("../../puresound/web/static/routes.js");

test("the screens, in navigation order", () => {
  assert.deepEqual(R.SCREENS, ["playground", "verify", "compare", "annotate", "world", "pipeline", "models", "history"]);
});

test("every screen resolves to itself and round-trips through its hash", () => {
  for (const screen of R.SCREENS) {
    assert.deepEqual(R.resolveRoute(R.hashFor(screen)), { screen, redirected: false });
    assert.equal(R.hashFor(screen), `#/${screen}`);
  }
});

test("addresses from before the redesign redirect", () => {
  assert.deepEqual(R.resolveRoute("#/zoo"), { screen: "models", redirected: true });
  assert.deepEqual(R.resolveRoute("#/measurements"), { screen: "compare", redirected: true });
  assert.deepEqual(R.resolveRoute("#/playground/voice"), { screen: "playground", redirected: true });
  assert.deepEqual(R.resolveRoute("#/playground/sv"), { screen: "verify", redirected: true });
});

test("a hash without a screen leaves the choice to the stored or default screen", () => {
  for (const hash of ["", "#", "#/", undefined, null]) assert.equal(R.resolveRoute(hash), null);
});

test("unknown screens and extra segments are not guessed", () => {
  assert.equal(R.resolveRoute("#/nowhere"), null);
  assert.equal(R.resolveRoute("#/playground/unknown"), null);
  assert.deepEqual(R.resolveRoute("#/models/"), { screen: "models", redirected: true });
  assert.deepEqual(R.resolveRoute("#models"), { screen: "models", redirected: true });
});
