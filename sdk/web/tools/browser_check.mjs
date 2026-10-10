// Playground's "This device" runs in real browsers, against a running
// `puresound web`: the page is cross-origin isolated, each model with a device
// build runs a sample, server-only models stay listed and marked, a failed or
// cancelled model download and a cancelled run recover, the output matches the
// server's on the same sample, and the labels follow the language switch.
import { chromium, firefox, webkit } from "playwright";
import assert from "node:assert/strict";

const base = process.env.PURESOUND_WEB_URL || "http://127.0.0.1:7861";
const engines = (process.env.PURESOUND_BROWSERS || "chromium,firefox,webkit").split(",");
const executablePath = process.env.PURESOUND_CHROMIUM_PATH || undefined;
const RUN_TIMEOUT = 120000;

async function openPlayground(context, errors, query = "") {
  const page = await context.newPage();
  page.setDefaultTimeout(30000);
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${base}/${query}#/playground`);
  await page.evaluate(() => localStorage.setItem("puresound.runner", "device"));
  await page.reload();
  await page.waitForFunction(() => document.querySelector('[data-runner="device"]').classList.contains("is-active"));
  await page.locator("[data-sample]").first().click();
  await page.waitForFunction(() => document.querySelector("#voice-dropzone").classList.contains("has-file"));
  return page;
}

// Run starts by clearing the note under the input; the note then says how it ended.
async function runAndWait(page) {
  await page.locator("#voice-run").click();
  await page.waitForFunction(() => /is-(success|error)/.test(document.querySelector("#voice-form-note").className), null, { timeout: RUN_TIMEOUT });
  return { ok: (await page.locator("#voice-form-note.is-success").count()) === 1, note: await page.locator("#voice-form-note").textContent() };
}

async function wavSamples(page, url) {
  return page.evaluate(async (href) => {
    const bytes = await (await fetch(href)).arrayBuffer();
    const view = new DataView(bytes);
    let offset = 12;
    while (String.fromCharCode(...new Uint8Array(bytes, offset, 4)) !== "data") offset += 8 + view.getUint32(offset + 4, true);
    return [...new Int16Array(bytes.slice(offset + 8, offset + 8 + view.getUint32(offset + 4, true)))];
  }, url);
}

for (const name of engines) {
  console.log(`Checking ${name}`);
  const browser = await { chromium, firefox, webkit }[name].launch({ headless: true, ...(name === "chromium" && executablePath ? { executablePath } : {}) });
  const errors = [];
  try {
    // A failed and a cancelled model download leave the page ready for the next run.
    const interrupted = await browser.newContext();
    const page = await openPlayground(interrupted, errors, "?threads=1");
    await page.evaluate(() => {
      window.originalFetch = window.fetch;
      window.fetch = (url, options) => (String(url).endsWith(".onnx") ? Promise.reject(new TypeError("Simulated interrupted download")) : window.originalFetch(url, options));
    });
    let run = await runAndWait(page);
    assert.ok(!run.ok && /Simulated interrupted download/.test(run.note), run.note);
    await page.evaluate(() => {
      window.fetch = (url, options) => {
        if (!String(url).endsWith(".onnx")) return window.originalFetch(url, options);
        window.downloadStarted = true;
        return new Promise((resolve, reject) => options.signal.addEventListener("abort", () => {
          window.downloadAborted = true;
          reject(new DOMException("Aborted", "AbortError"));
        }));
      };
    });
    await page.locator("#voice-run").click();
    await page.waitForFunction(() => window.downloadStarted);
    await page.locator("#voice-cancel").click();
    await page.waitForFunction(() => window.downloadAborted && !document.querySelector("#voice-run").disabled, null, { timeout: 3000 });
    assert.equal(await page.locator("#voice-form-note").textContent(), "Inference cancelled.");
    await page.evaluate(() => { window.fetch = window.originalFetch; });
    run = await runAndWait(page);
    assert.ok(run.ok, run.note);
    assert.match(await page.locator("#voice-measurement-note").textContent(), /WebAssembly · 1 thread\./);
    await interrupted.close();
    console.log(`${name}: failed download, cancelled download and retry passed`);

    const context = await browser.newContext();
    const main = await openPlayground(context, errors);
    assert.equal(await main.evaluate(() => crossOriginIsolated), true, "the app is cross-origin isolated");
    assert.match(await main.locator("#voice-runner-hint").textContent(), /WebAssembly, \d threads/);
    const options = await main.$$eval("#voice-model option", (items) => items.map((item) => ({ value: item.value, disabled: item.disabled, text: item.textContent })));
    const onDevice = options.filter((option) => !option.disabled);
    assert.ok(onDevice.length >= 1, "at least one model has a device build");
    for (const option of options.filter((item) => item.disabled)) assert.match(option.text, /server only$/);

    for (const option of onDevice) {
      await main.locator("#voice-model").selectOption(option.value);
      run = await runAndWait(main);
      assert.ok(run.ok, run.note);
      assert.ok((await wavSamples(main, await main.locator("#voice-download").getAttribute("href"))).length > 0);
      assert.match(await main.locator("#voice-measurement-note").textContent(), /^Processed on this device with WebAssembly · \d threads\./);
      await main.locator("#voice-deck .audio-play").click();
      await main.locator('#voice-deck [role="radio"][data-track="input"]').click();
      await main.locator("#voice-deck .audio-play").click();
      console.log(`${name} ${option.value}: ${(await main.locator("#voice-metrics").innerText()).replace(/\s+/g, " ")}`);
    }

    // Cancelling a run ends its worker; the next run loads the model again.
    await main.locator("#voice-run").click();
    await main.locator("#voice-cancel").click();
    await main.waitForFunction(() => !document.querySelector("#voice-run").disabled);
    assert.equal(await main.locator("#voice-form-note").textContent(), "Inference cancelled.");
    run = await runAndWait(main);
    assert.ok(run.ok, run.note);

    // The same sample on the server, then here: the server's output becomes the
    // Previous track, and the two line up sample for sample.
    await main.locator('[data-runner="server"]').click();
    run = await runAndWait(main);
    assert.ok(run.ok, run.note);
    const server = await wavSamples(main, await main.locator("#voice-download").getAttribute("href"));
    await main.locator('[data-runner="device"]').click();
    run = await runAndWait(main);
    assert.ok(run.ok, run.note);
    assert.equal(await main.locator('#voice-deck [role="radio"][data-track="previous"]').count(), 1);
    const device = await wavSamples(main, await main.locator("#voice-download").getAttribute("href"));
    let error = 0;
    let energy = 0;
    for (let index = 0; index < Math.min(server.length, device.length); index += 1) {
      error += (server[index] - device[index]) ** 2;
      energy += server[index] ** 2;
    }
    const nrms = Math.sqrt(error / energy);
    console.log(`${name}: device output against the server's, NRMS ${nrms.toExponential(2)}`);
    assert.ok(nrms < 2e-3, `NRMS ${nrms}`);

    await main.locator('[data-lang="zh-TW"]').first().click();
    assert.equal(await main.locator('[data-runner="device"]').textContent(), "這台裝置");
    assert.match(await main.locator("#voice-runner-hint").textContent(), /WebAssembly/);
    await main.locator('[data-lang="en"]').first().click();
    assert.equal(await main.locator('[data-runner="device"]').textContent(), "This device");
    assert.deepEqual(errors, [], `${name} page errors`);
    console.log(`${name}: device models, cancellation, server parity and language switch passed`);
    await context.close();
  } finally {
    await browser.close();
  }
}
