import { Client, handle_file } from "@gradio/client";
import { readFile } from "node:fs/promises";
import { basename } from "node:path";

const [workerUrl, archivePath] = process.argv.slice(2);
if (!workerUrl || !archivePath) {
  throw new Error("Usage: node scripts/verify-worker.mjs <worker-url> <case.zip>");
}

const client = await Client.connect(workerUrl, { events: ["data", "status"] });
const api = await client.view_api();
if (!api.named_endpoints?.["/analyze"]) {
  throw new Error("The worker does not expose the named /analyze endpoint.");
}

const archive = new File(
  [await readFile(archivePath)],
  basename(archivePath),
  { type: "application/zip" },
);
const job = client.submit("/analyze", [handle_file(archive)]);
let result = null;
for await (const message of job) {
  if (message.type === "status") {
    process.stdout.write(`stage=${message.stage}\n`);
    if (message.stage === "error") {
      throw new Error(message.message || "The worker reported an error.");
    }
  }
  if (message.type === "data") result = message.data;
}

if (!Array.isArray(result) || result.length < 4) {
  throw new Error("The worker returned an incomplete result.");
}

const manifest = typeof result[0] === "string" ? JSON.parse(result[0]) : result[0];
process.stdout.write(`${JSON.stringify({
  case_id: manifest.case_id,
  model_id: manifest.inference?.model_id,
  gpu: manifest.inference?.hardware,
  elapsed_seconds: manifest.inference?.elapsed_seconds,
  whole_tumor_ml: manifest.measurements?.regions?.whole_tumor?.volume_ml,
  artifacts: result.slice(1).map((item) => item?.url || item?.path || item),
}, null, 2)}\n`);
