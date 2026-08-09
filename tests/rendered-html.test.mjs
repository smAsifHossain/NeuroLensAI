import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html", host: "localhost" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the NeuroLens product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>NeuroLens AI/);
  assert.match(html, /Explore brain tumor regions/);
  assert.match(html, /Across every MRI sequence/);
  assert.match(html, /Upload MRI scan ZIP/);
  assert.match(html, /Download sample MRI scans/);
  assert.match(html, /github\.com\/smAsifHossain\/NeuroLensAI\/tree\/main\/sample-mri-scans/);
  assert.match(html, /SegResNet model/);
  assert.match(html, /Developed by S M Asif Hossain/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("shows results only after a real worker response", async () => {
  const [page, layout, packageJson, hosting] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../.openai/hosting.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /runAnalysis/);
  assert.match(page, /@gradio\/client/);
  assert.match(page, /\/api\/worker-config/);
  assert.match(page, /worker-config\.json/);
  assert.match(page, /GPU ONLINE/);
  assert.match(page, /Run brain tumor segmentation/);
  assert.match(page, /SAMPLE_DATA_URL/);
  assert.match(page, /five sample cases/);
  assert.match(page, /result &&/);
  assert.match(page, /GENERATED FROM THIS UPLOAD/);
  assert.match(page, /does not provide a diagnosis/i);
  assert.match(page, /Download PDF summary/);
  assert.match(page, /ONE ORIGINAL \+ FOUR AI OVERLAYS/);
  assert.match(page, /Normalize non-zero intensity values/);
  assert.match(page, /Explore the SegResNet architecture and processing pipeline/);
  assert.match(page, /No VLM is used for report writing/);
  assert.match(page, /SegResNet performance, with the evaluation context attached/);
  assert.match(page, /0\.9026/);
  assert.match(page, /0\.8559/);
  assert.match(page, /0\.7905/);
  assert.match(page, /0\.8518/);
  assert.doesNotMatch(page, /What these results do—and do not—establish|has not independently reproduced them/);
  assert.match(page, /A100 80 GB/);
  assert.doesNotMatch(page, /MONAI SegResNet|SegResNet mask|Clear AI segmentation/);
  assert.doesNotMatch(page, /Nothing is preloaded|Select BraTS ZIP|MRI RESEARCH WORKSTATION/);
  assert.doesNotMatch(page, /VERIFIED|BraTS-GLI-00002-000|108\.551|Explore verified result/);
  assert.match(layout, /openGraph/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  const hostingConfig = JSON.parse(hosting);
  assert.match(hostingConfig.project_id, /^appgprj_/);
  assert.equal(hostingConfig.d1, null);
  assert.equal(hostingConfig.r2, null);

  await access(new URL("../public/og.png", import.meta.url));
  await access(new URL("../LICENSE", import.meta.url));
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
  await access(new URL("../pnpm-lock.yaml", import.meta.url));
});

test("serves the managed inference worker configuration", async () => {
  process.env.NEUROLENS_WORKER_URL = "https://worker.example";
  const workerModule = new URL("../dist/server/index.js", import.meta.url);
  workerModule.searchParams.set("config-test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerModule.href);

  const response = await worker.fetch(
    new Request("http://localhost/api/worker-config", { headers: { host: "localhost" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { workerUrl: "https://worker.example" });
  assert.match(response.headers.get("cache-control") ?? "", /no-store/);
  delete process.env.NEUROLENS_WORKER_URL;
});
