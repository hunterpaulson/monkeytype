/* eslint-disable typescript-eslint/no-unsafe-assignment, typescript-eslint/no-unsafe-call, typescript-eslint/no-unsafe-member-access, typescript-eslint/no-unsafe-argument */
import { spawnSync } from "node:child_process";
import { createWriteStream, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { Readable } from "node:stream";
import { finished } from "node:stream/promises";

const bucket = process.env.TYPEGPT_WEIGHTS_BUCKET ?? "typegpt-weights";
const sourceBaseUrl =
  process.env.TYPEGPT_SOURCE_WEIGHTS_BASE_URL ??
  "https://raw.githubusercontent.com/hunterpaulson/webgpt-gpt2-weights/main/";

const weightKeys = buildGpt2WeightKeys();

for (const key of weightKeys) {
  const url = new URL(key, sourceBaseUrl).href;
  const tmpPath = join(tmpdir(), `typegpt-${basename(key)}`);

  console.log(`Downloading ${url}`);
  await downloadToFile(url, tmpPath);

  console.log(`Uploading r2://${bucket}/${key}`);
  const result = spawnSync(
    "npx",
    ["wrangler", "r2", "object", "put", `${bucket}/${key}`, "--file", tmpPath],
    { stdio: "inherit" },
  );
  rmSync(tmpPath, { force: true });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function buildGpt2WeightKeys() {
  const keys = [
    "weights/gpt2/params_gpt.json",
    "weights/gpt2/transformer.wte.weight_gpt.bin",
    "weights/gpt2/transformer.wpe.weight_gpt.bin",
    "weights/gpt2/transformer.ln_f.weight_gpt.bin",
    "weights/gpt2/transformer.ln_f.bias_gpt.bin",
  ];

  for (let layer = 0; layer < 12; layer++) {
    const prefix = `weights/gpt2/transformer.h.${layer}.`;
    keys.push(
      `${prefix}ln_1.weight_gpt.bin`,
      `${prefix}ln_1.bias_gpt.bin`,
      `${prefix}attn.c_attn.weight_gpt.bin`,
      `${prefix}attn.c_attn.bias_gpt.bin`,
      `${prefix}attn.c_proj.weight_gpt.bin`,
      `${prefix}attn.c_proj.bias_gpt.bin`,
      `${prefix}ln_2.weight_gpt.bin`,
      `${prefix}ln_2.bias_gpt.bin`,
      `${prefix}mlp.c_fc.weight_gpt.bin`,
      `${prefix}mlp.c_fc.bias_gpt.bin`,
      `${prefix}mlp.c_proj.weight_gpt.bin`,
      `${prefix}mlp.c_proj.bias_gpt.bin`,
    );
  }

  return keys;
}

async function downloadToFile(url, path) {
  const response = await fetch(url);
  if (!response.ok || response.body === null) {
    throw new Error(`Failed to download ${url}: ${response.status}`);
  }

  await finished(Readable.fromWeb(response.body).pipe(createWriteStream(path)));
}
