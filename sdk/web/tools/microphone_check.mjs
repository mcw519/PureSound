// Playground's microphone in "This device" mode on Chromium, with a simulated
// capture device: a recording runs on this device; a live session either keeps
// up and opens as a comparison when stopped, or is refused by the speed check
// with the recording route offered; a denied permission leaves the controls usable.
import { chromium } from "playwright";
import assert from "node:assert/strict";

const base = process.env.PURESOUND_WEB_URL || "http://127.0.0.1:7861";
const seconds = Number(process.env.PURESOUND_LIVE_SECONDS || 10);
const executablePath = process.env.PURESOUND_CHROMIUM_PATH || undefined;
const browser = await chromium.launch({
  headless: true,
  executablePath,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
});

async function openPlayground(context, errors) {
  const page = await context.newPage();
  page.setDefaultTimeout(30000);
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${base}/#/playground`);
  await page.evaluate(() => localStorage.setItem("puresound.runner", "device"));
  await page.reload();
  await page.waitForFunction(() => document.querySelector('[data-runner="device"]').classList.contains("is-active"));
  return page;
}

const isLive = (page) => page.locator("#live-toggle").evaluate((button) => button.classList.contains("is-recording"));

async function startLive(page) {
  await page.locator("#live-toggle").click();
  await page.waitForFunction(() => document.querySelector("#live-toggle").classList.contains("is-recording") || document.querySelector("#live-note").classList.contains("is-error"), null, { timeout: 120000 });
  return isLive(page);
}

try {
  const errors = [];
  const context = await browser.newContext({ permissions: ["microphone"] });
  const page = await openPlayground(context, errors);

  // The recording runs on a second model with a device build, live on the default one.
  const [first, second] = await page.$$eval("#voice-model option", (items) => items.filter((item) => !item.disabled).map((item) => item.value));
  if (second) await page.locator("#voice-model").selectOption(second);
  await page.locator('[data-input-mode="record"]').click();
  await page.locator("#voice-recorder .record-button").click();
  await page.waitForFunction(() => document.querySelector("#voice-recorder .recorder").classList.contains("is-recording"));
  await page.waitForTimeout(1000);
  await page.locator("#voice-recorder .record-button").click();
  await page.waitForFunction(() => document.querySelector("#voice-dropzone").classList.contains("has-file"));
  await page.locator("#voice-run").click();
  await page.waitForFunction(() => /is-(success|error)/.test(document.querySelector("#voice-form-note").className), null, { timeout: 120000 });
  assert.match(await page.locator("#voice-form-note").textContent(), /^Done/);
  assert.match(await page.locator("#voice-result-name").textContent(), /recording-.*this device$/);

  await page.locator('[data-input-mode="live"]').click();
  await page.locator("#voice-model").selectOption(first);
  if (!(await startLive(page))) {
    const reason = await page.locator("#live-note").textContent();
    assert.match(reason, /too slow to keep up live/);
    console.log("This device was refused live and offered recording instead:", reason);
    if (seconds >= 600) throw new Error("Ten-minute live acceptance needs a device whose measured RTF is at most 0.7");
  } else {
    await page.locator('[data-monitor="model"]').click();
    for (let elapsed = 0; elapsed < seconds; elapsed += 10) {
      await page.waitForTimeout(Math.min(10, seconds - elapsed) * 1000);
      if (!(await isLive(page))) {
        const reason = await page.locator("#live-note").textContent();
        assert.match(reason, /fell a second behind/);
        console.log("The bounded queue stopped an overloaded device:", reason);
        if (seconds >= 600) throw new Error("Ten-minute live acceptance remains unverified: the device fell behind");
        break;
      }
      console.log(`Live ${Math.min(elapsed + 10, seconds)} s: ${(await page.locator("#live-stats").innerText()).replace(/\s+/g, " ")}`);
    }
    if (await isLive(page)) await page.locator("#live-toggle").click();
    await page.waitForFunction(() => document.querySelector("#voice-result-name").textContent.startsWith("Live session"), null, { timeout: 120000 });
    assert.match(await page.locator("#voice-result-name").textContent(), /this device$/);
    // A new session starts from a fresh stream.
    if (await startLive(page)) {
      await page.waitForTimeout(500);
      await page.locator("#live-toggle").click();
      await page.waitForFunction(() => !document.querySelector("#live-toggle").classList.contains("is-recording"));
    }
  }
  assert.deepEqual(errors, []);

  // A denied microphone is reported and leaves recording available.
  const denied = await browser.newContext({ permissions: [] });
  const refused = await openPlayground(denied, errors);
  await refused.evaluate(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException("Permission denied", "NotAllowedError"); };
  });
  await refused.locator('[data-input-mode="record"]').click();
  await refused.locator("#voice-recorder .record-button").click();
  await refused.waitForFunction(() => document.querySelector("#toast").textContent.includes("Permission denied"));
  assert.equal(await refused.locator("#voice-recorder .record-button").isEnabled(), true);
  assert.equal(await refused.locator("#voice-recorder .recorder.is-recording").count(), 0);
  assert.deepEqual(errors, []);
  console.log("Recording on this device, live, its speed check and a denied microphone passed");
  await context.close();
  await denied.close();
} finally {
  await browser.close();
}
