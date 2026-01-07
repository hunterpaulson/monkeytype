/**
 * End-to-end benchmark for constrained decoding performance.
 *
 * Usage:
 *   cd frontend && npm exec tsx -- src/ts/test/llm/run-benchmark.ts
 *
 * This script:
 * 1. Loads the GPT-2 model
 * 2. Builds the precomputed constraint processor
 * 3. Generates words with constrained decoding
 * 4. Reports timing breakdown for each phase
 */

import { env, pipeline, TextGenerationPipeline } from "@huggingface/transformers";
import { PrecomputedConstrainedProcessor } from "./precomputed-constrained-processor";
import { BenchmarkContext, Timer } from "./benchmark";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

// Configure transformers.js for Node.js
env.useBrowserCache = false;
env.allowLocalModels = false;

// ESM equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load wordset from actual language file
function loadWordset(languageFile: string = "english.json"): string[] {
  // Try multiple possible paths
  const possiblePaths = [
    path.resolve(__dirname, "../../../static/languages", languageFile),
    path.resolve(process.cwd(), "static/languages", languageFile),
    path.resolve(process.cwd(), "frontend/static/languages", languageFile),
  ];
  
  for (const filePath of possiblePaths) {
    try {
      const content = fs.readFileSync(filePath, "utf-8");
      const data = JSON.parse(content) as { words: string[] };
      console.log(`Loaded wordset from: ${filePath}`);
      return data.words;
    } catch {
      // Try next path
    }
  }
  
  console.warn(`Could not load ${languageFile} from any path, using fallback wordset`);
  // Fallback to common words if file not found
  return [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "from", "or", "which", "one", "would", "all", "will", "there", "say",
    "who", "make", "when", "can", "more", "if", "no", "man", "out", "other",
    "so", "what", "time", "up", "go", "about", "than", "into", "could", "state",
  ];
}

const MODEL_ID = "Xenova/distilgpt2";
const TARGET_WORDS = 100;
const MAX_NEW_TOKENS = 500;

async function runBenchmark(languageFile: string = "english.json"): Promise<void> {
  const wordset = loadWordset(languageFile);
  
  console.log("=".repeat(70));
  console.log("CONSTRAINED DECODING BENCHMARK");
  console.log("=".repeat(70));
  console.log(`\nModel: ${MODEL_ID}`);
  console.log(`Language: ${languageFile}`);
  console.log(`Wordset: ${wordset.length} words`);
  console.log(`Target: ${TARGET_WORDS} words`);

  const benchmark = new BenchmarkContext();
  benchmark.setWordsetSize(wordset.length);

  // Phase 1: Load model
  console.log("\n--- Phase 1: Loading Model ---");
  const modelTimer = new Timer();
  modelTimer.start();

  const gen = await pipeline("text-generation", MODEL_ID);
  const generator = gen as TextGenerationPipeline;

  const modelLoadTime = modelTimer.stop();
  console.log(`Model loaded in ${modelLoadTime.toFixed(0)}ms`);

  // Get tokenizer and vocab size
  const tokenizer = (generator as unknown as { tokenizer?: unknown }).tokenizer;
  const vocabSize =
    (generator as unknown as { model?: { config?: { vocab_size?: number } } }).model
      ?.config?.vocab_size ?? 50257;

  if (!tokenizer) {
    throw new Error("Could not access tokenizer");
  }

  console.log(`Vocabulary size: ${vocabSize}`);
  benchmark.setVocabSize(vocabSize);

  // Phase 2: Build constraint processor
  console.log("\n--- Phase 2: Building Constraint Processor ---");
  benchmark.startInit();

  const processor = new PrecomputedConstrainedProcessor(
    wordset,
    tokenizer as {
      decode: (tokens: Array<number | bigint>, options?: { skip_special_tokens?: boolean }) => string;
    },
    vocabSize,
    {
      benchmark,
      onProgress: (progress) => {
        if (progress.percent % 25 === 0 || progress.phase === "ready") {
          console.log(`  [${progress.phase}] ${progress.percent}% - ${progress.detail ?? ""}`);
        }
      },
    }
  );

  benchmark.endInit();
  const initResult = benchmark.getInitTiming();
  console.log(`\nConstraint processor built:`);
  console.log(`  Token cache: ${initResult.tokenCacheMs.toFixed(0)}ms`);
  console.log(`  State machine: ${initResult.stateMachineBuildMs.toFixed(0)}ms`);
  console.log(`  Total: ${initResult.totalMs.toFixed(0)}ms`);

  const stats = processor.getStats();
  console.log(`  States: ${stats.stateCount}`);
  console.log(`  Avg valid tokens/state: ${stats.avgValidTokens.toFixed(1)}`);

  // Phase 3: Generate words
  console.log("\n--- Phase 3: Generating Words ---");
  benchmark.startGeneration();

  const genTimer = new Timer();
  genTimer.start();

  let allWords: string[] = [];
  let totalTokens = 0;
  let batchCount = 0;

  while (allWords.length < TARGET_WORDS) {
    batchCount++;
    processor.reset();

    const options: Record<string, unknown> = {
      max_new_tokens: MAX_NEW_TOKENS,
      temperature: 0.8,
      top_p: 0.9,
      do_sample: true,
      return_full_text: false,
      logits_processor: [processor.getProcessor()],
    };

    // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-assignment
    const outputs = await generator(" ", options);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    const generated = (outputs as Array<{ generated_text?: string }>)?.[0]?.generated_text;

    if (generated) {
      const words = generated
        .split(/\s+/)
        .map((w) => w.trim().toLowerCase())
        .filter((w) => w.length > 0 && /^[a-z]+$/.test(w));

      allWords.push(...words);
      totalTokens += MAX_NEW_TOKENS; // Approximate

      console.log(`  Batch ${batchCount}: +${words.length} words (total: ${allWords.length})`);
    }
  }

  const genTime = genTimer.stop();
  benchmark.endGeneration();

  // Trim to target
  allWords = allWords.slice(0, TARGET_WORDS);

  console.log(`\nGenerated ${allWords.length} words in ${genTime.toFixed(0)}ms`);
  console.log(`  Time per word: ${(genTime / allWords.length).toFixed(1)}ms`);
  console.log(`  Batches: ${batchCount}`);

  // Validate all words are in wordset
  const wordsetSet = new Set(wordset.map((w) => w.toLowerCase()));
  const invalidWords = allWords.filter((w) => !wordsetSet.has(w));

  if (invalidWords.length > 0) {
    console.log(`\n⚠️  WARNING: ${invalidWords.length} invalid words found!`);
    console.log(`  Invalid: ${invalidWords.slice(0, 10).join(", ")}${invalidWords.length > 10 ? "..." : ""}`);
  } else {
    console.log(`\n✅ All ${allWords.length} words are valid!`);
  }

  // Show sample
  console.log(`\n--- Sample Output ---`);
  console.log(allWords.slice(0, 30).join(" "));

  // Final summary
  console.log("\n" + "=".repeat(70));
  console.log("BENCHMARK SUMMARY");
  console.log("=".repeat(70));
  console.log(`\nInitialization:`);
  console.log(`  Model load:       ${modelLoadTime.toFixed(0)}ms`);
  console.log(`  Token cache:      ${initResult.tokenCacheMs.toFixed(0)}ms`);
  console.log(`  State machine:    ${initResult.stateMachineBuildMs.toFixed(0)}ms`);
  console.log(`  Total init:       ${(modelLoadTime + initResult.totalMs).toFixed(0)}ms`);

  console.log(`\nGeneration:`);
  console.log(`  Words generated:  ${allWords.length}`);
  console.log(`  Total time:       ${genTime.toFixed(0)}ms`);
  console.log(`  Time per word:    ${(genTime / allWords.length).toFixed(1)}ms`);

  console.log(`\nState Machine:`);
  console.log(`  States:           ${stats.stateCount}`);
  console.log(`  Avg valid tokens: ${stats.avgValidTokens.toFixed(1)}`);

  console.log("\n" + "=".repeat(70));
}

// Run if called directly
if (typeof process !== "undefined") {
  runBenchmark()
    .then(() => {
      console.log("\n✅ Benchmark complete");
      process.exit(0);
    })
    .catch((error) => {
      console.error("\n❌ Benchmark failed:", error);
      process.exit(1);
    });
}

export { runBenchmark };

