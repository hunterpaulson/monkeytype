/**
 * Token-level profiling for constrained decoding.
 *
 * Measures:
 * - Time to first token (TTFT)
 * - Inter-token latency for each token
 * - Total decode steps
 * - KV cache usage
 *
 * Usage:
 *   cd frontend && npm exec tsx -- src/ts/test/llm/profile-tokens.ts
 */

import {
  env,
  AutoTokenizer,
  AutoModelForCausalLM,
  Tensor,
  type PreTrainedTokenizer,
  type PreTrainedModel,
} from "@huggingface/transformers";
import { PrecomputedConstrainedProcessor } from "./precomputed-constrained-processor";
import { Timer } from "./benchmark";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

// ESM __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configure for Node.js
env.useBrowserCache = false;
env.allowLocalModels = false;

const MODEL_ID = "Xenova/distilgpt2";

// Check for WebGPU availability
async function checkWebGPU(): Promise<boolean> {
  try {
    // @ts-expect-error - navigator.gpu typing
    if (typeof navigator !== "undefined" && navigator.gpu) {
      // @ts-expect-error - navigator.gpu typing
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        console.log("[WebGPU] Available!");
        return true;
      }
    }
  } catch (e) {
    console.log("[WebGPU] Not available:", e);
  }
  console.log("[WebGPU] Not available, using WASM/CPU");
  return false;
}

// Load wordset
function loadWordset(): string[] {
  const possiblePaths = [
    path.resolve(__dirname, "../../../static/languages/english.json"),
    path.resolve(process.cwd(), "static/languages/english.json"),
    path.resolve(process.cwd(), "frontend/static/languages/english.json"),
  ];

  for (const filePath of possiblePaths) {
    try {
      const content = fs.readFileSync(filePath, "utf-8");
      const data = JSON.parse(content) as { words: string[] };
      return data.words;
    } catch {
      // Try next
    }
  }

  return ["the", "be", "to", "of", "and", "a", "in", "that", "have", "it"];
}

interface TokenTiming {
  tokenId: number;
  tokenStr: string;
  forwardPassMs: number;
  constraintCheckMs: number;
  samplingMs: number;
  totalMs: number;
  cumulativeMs: number;
  kvCacheSize: number;
}

// Type for KV cache from transformers.js
type KVCache = Array<{
  key: Tensor;
  value: Tensor;
}>;

/**
 * Extract KV cache from model output.
 * transformers.js returns KV cache as present.X.key/value flat structure
 */
function extractKVCache(outputs: Record<string, Tensor>): KVCache | null {
  // Check if this is the flat format (present.0.key, present.0.value, etc.)
  const layers: KVCache = [];
  let layerIdx = 0;
  
  while (true) {
    const keyTensor = outputs[`present.${layerIdx}.key`];
    const valueTensor = outputs[`present.${layerIdx}.value`];
    
    if (!keyTensor || !valueTensor) {
      break;
    }
    
    layers.push({ key: keyTensor, value: valueTensor });
    layerIdx++;
  }
  
  // Also check for past_key_values format
  if (layers.length === 0 && outputs["past_key_values"]) {
    return outputs["past_key_values"] as unknown as KVCache;
  }
  
  return layers.length > 0 ? layers : null;
}

/**
 * Convert KV cache to the format expected by model.forward()
 * Model expects: { past_key_values.0.key, past_key_values.0.value, ... }
 */
function kvCacheToModelInput(cache: KVCache): Record<string, Tensor> {
  const result: Record<string, Tensor> = {};
  
  cache.forEach((layer, idx) => {
    result[`past_key_values.${idx}.key`] = layer.key;
    result[`past_key_values.${idx}.value`] = layer.value;
  });
  
  return result;
}

/**
 * Trim KV cache by removing the oldest N entries (for sliding window).
 * KV cache shape per layer: [batch, num_heads, seq_len, head_dim]
 */
function trimKVCache(cache: KVCache, trimCount: number): KVCache {
  return cache.map((layer) => {
    const { key, value } = layer;
    
    // Get dimensions: [batch, num_heads, seq_len, head_dim]
    const seqLen = key.dims[2] as number;
    const newSeqLen = seqLen - trimCount;
    
    if (newSeqLen <= 0) {
      const emptyDims = [...key.dims];
      emptyDims[2] = 0;
      return {
        key: new Tensor(key.type, new Float32Array(0), emptyDims),
        value: new Tensor(value.type, new Float32Array(0), emptyDims),
      };
    }

    // Slice out the oldest entries
    const batch = key.dims[0] as number;
    const numHeads = key.dims[1] as number;
    const headDim = key.dims[3] as number;
    
    const keyData = key.data as Float32Array;
    const valueData = value.data as Float32Array;
    
    const newKeyData = new Float32Array(batch * numHeads * newSeqLen * headDim);
    const newValueData = new Float32Array(batch * numHeads * newSeqLen * headDim);

    // Copy data, skipping first trimCount positions
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < numHeads; h++) {
        for (let s = 0; s < newSeqLen; s++) {
          const oldSeqIdx = s + trimCount;
          for (let d = 0; d < headDim; d++) {
            const oldIdx = ((b * numHeads + h) * seqLen + oldSeqIdx) * headDim + d;
            const newIdx = ((b * numHeads + h) * newSeqLen + s) * headDim + d;
            newKeyData[newIdx] = keyData[oldIdx]!;
            newValueData[newIdx] = valueData[oldIdx]!;
          }
        }
      }
    }

    return {
      key: new Tensor(key.type, newKeyData, [batch, numHeads, newSeqLen, headDim]),
      value: new Tensor(value.type, newValueData, [batch, numHeads, newSeqLen, headDim]),
    };
  });
}

interface ProfilingResult {
  modelLoadMs: number;
  processorBuildMs: number;
  ttftMs: number; // Time to first token
  avgInterTokenMs: number;
  totalTokens: number;
  totalWords: number;
  totalTimeMs: number;
  tokenTimings: TokenTiming[];
  generatedText: string;
  kvCacheUsed: boolean;
}

async function profileGeneration(
  targetWords: number = 50,
  contextWindowSize: number | null = null // null = unlimited
): Promise<ProfilingResult> {
  console.log("=".repeat(70));
  console.log("TOKEN-LEVEL PROFILING");
  console.log("=".repeat(70));
  console.log(`\nTarget words: ${targetWords}`);
  console.log(`Context window: ${contextWindowSize ?? "unlimited"}`);

  const wordset = loadWordset();
  console.log(`Wordset: ${wordset.length} words`);

  // Check WebGPU availability
  const hasWebGPU = await checkWebGPU();

  // Load model and tokenizer separately for manual generation
  console.log("\n--- Loading Model ---");
  const modelTimer = new Timer();
  modelTimer.start();

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  
  // In Node.js, use CPU (native ONNX runtime) - "wasm" and "webgpu" are browser-only
  // Browser would use: hasWebGPU ? "webgpu" : "wasm"
  const device = "cpu" as const;
  
  console.log(`Loading model with device: ${device}`);
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: "fp32",
    device,
  });

  const modelLoadMs = modelTimer.stop();
  console.log(`Model loaded in ${modelLoadMs.toFixed(0)}ms`);
  console.log(`Device: ${hasWebGPU ? "WebGPU" : "WASM/CPU"}`);

  // Get vocab size
  const vocabSize = (model.config as { vocab_size?: number }).vocab_size ?? 50257;
  console.log(`Vocab size: ${vocabSize}`);

  // Build constraint processor
  console.log("\n--- Building Constraint Processor ---");
  const processorTimer = new Timer();
  processorTimer.start();

  const processor = new PrecomputedConstrainedProcessor(
    wordset,
    tokenizer as unknown as {
      decode: (tokens: Array<number | bigint>, options?: { skip_special_tokens?: boolean }) => string;
    },
    vocabSize
  );

  const processorBuildMs = processorTimer.stop();
  console.log(`Processor built in ${processorBuildMs.toFixed(0)}ms`);
  console.log(`States: ${processor.getStats().stateCount}`);

  // Manual token-by-token generation
  console.log("\n--- Generating Tokens ---");

  const tokenTimings: TokenTiming[] = [];
  let generatedIds: bigint[] = [];
  let pastKeyValues: unknown = null;
  let kvCacheUsed = false;

  // Start with BOS token or space
  const startTokens = tokenizer.encode(" ", { add_special_tokens: false });
  generatedIds = startTokens.map((t: number) => BigInt(t));

  const totalTimer = new Timer();
  totalTimer.start();

  let tokenCount = 0;
  let wordCount = 0;
  const maxTokens = targetWords * 6; // Estimate ~6 tokens per word max

  // Track cached token count for sliding window with KV cache
  let cachedTokenCount = 0;

  while (wordCount < targetWords && tokenCount < maxTokens) {
    const tokenTimer = new Timer();
    tokenTimer.start();

    // Prepare input based on whether we have KV cache
    let inputIds: bigint[];
    let positionOffset = 0;
    let useCache = true; // Always try to use cache
    let trimCache = false;

    if (pastKeyValues === null) {
      // First call - no cache yet
      // If we have a context window limit, only take last N tokens
      if (contextWindowSize !== null && generatedIds.length > contextWindowSize) {
        inputIds = generatedIds.slice(-contextWindowSize);
      } else {
        inputIds = generatedIds;
      }
      cachedTokenCount = 0;
    } else if (contextWindowSize !== null && cachedTokenCount >= contextWindowSize) {
      // Sliding window with KV cache - need to trim the oldest entry
      inputIds = [generatedIds[generatedIds.length - 1]!];
      positionOffset = contextWindowSize - 1; // Position after trim
      trimCache = true;
    } else {
      // Incremental decoding - only pass the last token
      inputIds = [generatedIds[generatedIds.length - 1]!];
      positionOffset = cachedTokenCount;
    }

    // Trim KV cache if needed (sliding window)
    if (trimCache && pastKeyValues) {
      pastKeyValues = trimKVCache(pastKeyValues as KVCache, 1);
      cachedTokenCount--;
    }

    // Convert to tensor
    const inputTensor = new Tensor(
      "int64",
      inputIds,
      [1, inputIds.length]
    );

    // Create attention mask - for incremental, we need to account for past tokens
    const totalSeqLen = pastKeyValues ? positionOffset + inputIds.length : inputIds.length;
    const attentionMask = new Tensor(
      "int64",
      new BigInt64Array(totalSeqLen).fill(BigInt(1)),
      [1, totalSeqLen]
    );

    // Create position ids
    const positionIds = new Tensor(
      "int64",
      inputIds.map((_, i) => BigInt(positionOffset + i)),
      [1, inputIds.length]
    );

    // Forward pass
    const forwardTimer = new Timer();
    forwardTimer.start();

    // Build forward inputs - spread KV cache entries individually
    const forwardInputs: Record<string, unknown> = {
      input_ids: inputTensor,
      attention_mask: attentionMask,
      position_ids: positionIds,
      use_cache: useCache,
    };
    
    // Add KV cache if available (transformers.js expects past_key_values.X.key/value format)
    if (pastKeyValues && useCache) {
      const kvInputs = kvCacheToModelInput(pastKeyValues as KVCache);
      Object.assign(forwardInputs, kvInputs);
    }

    const outputs = await model.forward(forwardInputs);

    const forwardPassMs = forwardTimer.stop();

    // Get logits for last token
    const logits = outputs.logits;
    const logitsData = logits.data as Float32Array;
    const lastTokenLogits = logitsData.slice(-vocabSize);

    // Apply constraints
    const constraintTimer = new Timer();
    constraintTimer.start();

    // Create logits tensor for processor
    const logitsTensor = {
      data: new Float32Array(lastTokenLogits),
      dims: [1, vocabSize],
    };

    processor.process([generatedIds], logitsTensor);

    const constraintCheckMs = constraintTimer.stop();

    // Sample next token
    const samplingTimer = new Timer();
    samplingTimer.start();

    const temperature = 0.8;
    const maskedLogits = logitsTensor.data;

    // Find valid tokens and apply softmax
    const validIndices: number[] = [];
    const validLogits: number[] = [];

    for (let i = 0; i < vocabSize; i++) {
      if (maskedLogits[i] !== -Infinity && !isNaN(maskedLogits[i]!)) {
        validIndices.push(i);
        validLogits.push(maskedLogits[i]! / temperature);
      }
    }

    if (validIndices.length === 0) {
      console.error("No valid tokens!");
      break;
    }

    // Softmax
    const maxLogit = Math.max(...validLogits);
    const expLogits = validLogits.map((l) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((e) => e / sumExp);

    // Sample
    let r = Math.random();
    let sampledIdx = 0;
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i]!;
      if (r <= 0) {
        sampledIdx = i;
        break;
      }
    }

    const nextTokenId = validIndices[sampledIdx]!;
    const samplingMs = samplingTimer.stop();

    // Update KV cache if available
    // transformers.js returns KV cache as present.X.key/value instead of past_key_values
    const newKVCache = extractKVCache(outputs);
    if (newKVCache) {
      pastKeyValues = newKVCache;
      cachedTokenCount += inputIds.length;
      kvCacheUsed = true;
    }

    // Decode token
    const tokenStr = tokenizer.decode([nextTokenId], { skip_special_tokens: false }) as string;

    // Add to sequence
    generatedIds.push(BigInt(nextTokenId));

    const totalTokenMs = tokenTimer.stop();

    // Record timing
    const kvCacheSize = pastKeyValues
      ? estimateKVCacheSize(pastKeyValues)
      : 0;

    tokenTimings.push({
      tokenId: nextTokenId,
      tokenStr,
      forwardPassMs,
      constraintCheckMs,
      samplingMs,
      totalMs: totalTokenMs,
      cumulativeMs: totalTimer.elapsed(),
      kvCacheSize,
    });

    // Check if we completed a word (token contains space or we're at word boundary)
    const currentText = tokenizer.decode(
      generatedIds.map((id) => Number(id)),
      { skip_special_tokens: true }
    ) as string;
    const words = currentText.trim().split(/\s+/).filter((w) => w.length > 0);
    wordCount = words.length;

    tokenCount++;

    // Progress
    if (tokenCount % 20 === 0) {
      console.log(
        `  Token ${tokenCount}: ${wordCount} words, ` +
        `${totalTokenMs.toFixed(1)}ms (fwd: ${forwardPassMs.toFixed(1)}, ` +
        `cstr: ${constraintCheckMs.toFixed(2)}, samp: ${samplingMs.toFixed(2)})`
      );
    }
  }

  const totalTimeMs = totalTimer.stop();

  // Decode final text
  const generatedText = tokenizer.decode(
    generatedIds.map((id) => Number(id)),
    { skip_special_tokens: true }
  ) as string;

  const finalWords = generatedText.trim().split(/\s+/).filter((w) => w.length > 0);

  // Calculate stats
  const ttftMs = tokenTimings[0]?.totalMs ?? 0;
  const interTokenTimes = tokenTimings.slice(1).map((t) => t.totalMs);
  const avgInterTokenMs =
    interTokenTimes.length > 0
      ? interTokenTimes.reduce((a, b) => a + b, 0) / interTokenTimes.length
      : 0;

  return {
    modelLoadMs,
    processorBuildMs,
    ttftMs,
    avgInterTokenMs,
    totalTokens: tokenCount,
    totalWords: finalWords.length,
    totalTimeMs,
    tokenTimings,
    generatedText,
    kvCacheUsed,
  };
}

function estimateKVCacheSize(pastKeyValues: unknown): number {
  // Rough estimate of KV cache size in bytes
  try {
    if (Array.isArray(pastKeyValues)) {
      let totalSize = 0;
      for (const layer of pastKeyValues) {
        if (layer && typeof layer === "object") {
          // Each layer has key and value tensors
          const layerObj = layer as { key?: { data?: Float32Array }, value?: { data?: Float32Array } };
          const keyData = layerObj.key?.data;
          const valueData = layerObj.value?.data;
          if (keyData) totalSize += keyData.length * 4; // float32
          if (valueData) totalSize += valueData.length * 4;
        }
      }
      return totalSize;
    }
  } catch {
    // Ignore errors
  }
  return 0;
}

function printResults(result: ProfilingResult): void {
  console.log("\n" + "=".repeat(70));
  console.log("PROFILING RESULTS");
  console.log("=".repeat(70));

  console.log("\n--- Timing Breakdown ---");
  console.log(`Model load:          ${result.modelLoadMs.toFixed(0)}ms`);
  console.log(`Processor build:     ${result.processorBuildMs.toFixed(0)}ms`);
  console.log(`Time to first token: ${result.ttftMs.toFixed(1)}ms`);
  console.log(`Avg inter-token:     ${result.avgInterTokenMs.toFixed(1)}ms`);
  console.log(`Total generation:    ${result.totalTimeMs.toFixed(0)}ms`);

  console.log("\n--- Token Stats ---");
  console.log(`Total tokens:        ${result.totalTokens}`);
  console.log(`Total words:         ${result.totalWords}`);
  console.log(`Tokens per word:     ${(result.totalTokens / result.totalWords).toFixed(2)}`);
  console.log(`KV cache used:       ${result.kvCacheUsed ? "YES" : "NO"}`);

  // Breakdown by component
  const avgForward =
    result.tokenTimings.reduce((a, t) => a + t.forwardPassMs, 0) / result.tokenTimings.length;
  const avgConstraint =
    result.tokenTimings.reduce((a, t) => a + t.constraintCheckMs, 0) / result.tokenTimings.length;
  const avgSampling =
    result.tokenTimings.reduce((a, t) => a + t.samplingMs, 0) / result.tokenTimings.length;

  console.log("\n--- Per-Token Breakdown (avg) ---");
  console.log(`Forward pass:        ${avgForward.toFixed(2)}ms (${((avgForward / result.avgInterTokenMs) * 100).toFixed(1)}%)`);
  console.log(`Constraint check:    ${avgConstraint.toFixed(3)}ms (${((avgConstraint / result.avgInterTokenMs) * 100).toFixed(2)}%)`);
  console.log(`Sampling:            ${avgSampling.toFixed(3)}ms (${((avgSampling / result.avgInterTokenMs) * 100).toFixed(2)}%)`);

  console.log("\n--- First 10 Token Timings ---");
  for (let i = 0; i < Math.min(10, result.tokenTimings.length); i++) {
    const t = result.tokenTimings[i]!;
    console.log(
      `  ${i + 1}. "${t.tokenStr.padEnd(12)}" ` +
      `total: ${t.totalMs.toFixed(1).padStart(6)}ms, ` +
      `fwd: ${t.forwardPassMs.toFixed(1).padStart(5)}ms, ` +
      `cstr: ${t.constraintCheckMs.toFixed(2).padStart(5)}ms`
    );
  }

  console.log("\n--- Generated Text (first 50 words) ---");
  const words = result.generatedText.trim().split(/\s+/);
  console.log(words.slice(0, 50).join(" "));

  console.log("\n" + "=".repeat(70));
}

// Main
async function main(): Promise<void> {
  // Profile without context window (uses KV cache)
  console.log("\n\n### PROFILE: No context window (KV cache enabled) ###\n");
  const result1 = await profileGeneration(50, null);
  printResults(result1);

  // Profile with context window (no KV cache benefit)
  console.log("\n\n### PROFILE: Context window = 10 tokens ###\n");
  const result2 = await profileGeneration(50, 10);
  printResults(result2);
}

main()
  .then(() => {
    console.log("\n✅ Profiling complete");
    process.exit(0);
  })
  .catch((error) => {
    console.error("\n❌ Profiling failed:", error);
    process.exit(1);
  });

