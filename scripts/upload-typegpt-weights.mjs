/* eslint-disable typescript-eslint/no-unsafe-assignment, typescript-eslint/no-unsafe-call, typescript-eslint/no-unsafe-member-access, typescript-eslint/no-unsafe-argument */
import { spawnSync } from "node:child_process";
import { createWriteStream, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { Readable } from "node:stream";
import { finished } from "node:stream/promises";

const bucket = process.env.TYPEGPT_WEIGHTS_BUCKET ?? "typegpt-weights";
const sourceRawBaseUrl =
  process.env.TYPEGPT_SOURCE_WEIGHTS_BASE_URL ??
  "https://raw.githubusercontent.com/hunterpaulson/webgpt-gpt2-weights/main/";
const sourceMediaBaseUrl =
  process.env.TYPEGPT_SOURCE_WEIGHTS_MEDIA_BASE_URL ??
  "https://media.githubusercontent.com/media/hunterpaulson/webgpt-gpt2-weights/main/";

const weightKeys = buildGpt2WeightKeys();

for (const key of weightKeys) {
  const sourceBaseUrl = key.endsWith(".bin")
    ? sourceMediaBaseUrl
    : sourceRawBaseUrl;
  const url = new URL(key, sourceBaseUrl).href;
  const tmpPath = join(tmpdir(), `typegpt-${basename(key)}`);

  console.log(`Downloading ${url}`);
  await downloadToFile(url, tmpPath);
  if (key.endsWith(".bin")) {
    assertNotGitLfsPointer(tmpPath, url);
  }

  console.log(`Uploading r2://${bucket}/${key}`);
  const result = spawnSync(
    "npx",
    [
      "wrangler",
      "r2",
      "object",
      "put",
      `${bucket}/${key}`,
      "--file",
      tmpPath,
      "--remote",
    ],
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

function assertNotGitLfsPointer(path, url) {
  const result = spawnSync("head", ["-c", "64", path], { encoding: "utf8" });

  if (result.stdout.startsWith("version https://git-lfs.github.com/spec/")) {
    throw new Error(
      `Downloaded Git LFS pointer instead of binary data: ${url}`,
    );
  }
}
