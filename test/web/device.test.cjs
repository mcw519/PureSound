// On-device inference's pure helpers (device/device.js), run by test_web_js.py.
const test = require("node:test");
const assert = require("node:assert/strict");
const D = require("../../puresound/web/static/device/device.js");

test("threads only on a cross-origin isolated page, at most four, and ?threads=1 forces one", () => {
  assert.equal(D.threadCount({ isolated: false, cores: 16 }), 1);
  assert.equal(D.threadCount({ isolated: true, cores: 16 }), 4);
  assert.equal(D.threadCount({ isolated: true, cores: 2 }), 2);
  assert.equal(D.threadCount({ isolated: true, cores: undefined }), 1);
  assert.equal(D.threadCount({ isolated: true, cores: 16, forced: 1 }), 1);
});

const MANIFEST = Object.freeze({
  hop_length: 160,
  streaming_delay_frames: 3,
  recommended_inference: Object.freeze({ dry_blend: 0.9, spec_floor: 0, mix_phase: false }),
  onset_guard: Object.freeze({ t_arm_s: 1, init_s: 0.2 }),
});

test("a run's settings override the manifest without changing it", () => {
  const out = D.configuredManifest(MANIFEST, { dryBlend: 0.5, onsetGuard: { t_arm_s: 2, margin_db: 10 } });
  assert.equal(out.recommended_inference.dry_blend, 0.5);
  assert.equal(out.recommended_inference.spec_floor, 0);
  assert.deepEqual(out.onset_guard, { t_arm_s: 2, init_s: 0.2, margin_db: 10 });
  assert.equal(MANIFEST.recommended_inference.dry_blend, 0.9);
  assert.equal(MANIFEST.onset_guard.t_arm_s, 1);
});

test("the onset guard can be switched off, switched on with defaults, or left to the manifest", () => {
  assert.equal("onset_guard" in D.configuredManifest(MANIFEST, { onsetGuard: false }), false);
  assert.deepEqual(D.configuredManifest({ ...MANIFEST, onset_guard: undefined }, { onsetGuard: true }).onset_guard, {});
  assert.deepEqual(D.configuredManifest(MANIFEST, {}).onset_guard, MANIFEST.onset_guard);
  assert.equal(D.configuredManifest(MANIFEST).recommended_inference.dry_blend, 0.9);
});

test("a dry blend outside (0, 1] is refused", () => {
  for (const dryBlend of [0, -0.1, 1.01, Number.NaN]) assert.throws(() => D.configuredManifest(MANIFEST, { dryBlend }), RangeError);
  assert.equal(D.configuredManifest(MANIFEST, { dryBlend: 1 }).recommended_inference.dry_blend, 1);
});

test("the emitted stream lines up with the input: look-ahead dropped, length kept", () => {
  const stream = Float32Array.from({ length: 12 }, (_, index) => index);
  assert.deepEqual([...D.alignStream(stream, 8, 3)], [3, 4, 5, 6, 7, 8, 9, 10]);
  // A stream that ends early is padded with silence to the input's length.
  assert.deepEqual([...D.alignStream(stream, 10, 4)], [4, 5, 6, 7, 8, 9, 10, 11, 0, 0]);
  assert.deepEqual([...D.alignStream(stream.subarray(0, 2), 3, 5)], [0, 0, 0]);
});
