/**
 * Debug script for constrained generation
 * 
 * Usage:
 *   npx tsx frontend/src/ts/test/llm/debug-constrained-generation.ts
 * 
 * Or with node:
 *   npm run build
 *   node dist/frontend/src/ts/test/llm/debug-constrained-generation.js
 */

import { pipeline, TextGenerationPipeline, env } from "@huggingface/transformers";
import { WordsetConstrainedLogitsProcessor } from "./constrained-logits-processor";

// Configure transformers.js
env.useBrowserCache = false; // Don't use browser cache in Node.js
env.allowLocalModels = false;

// Simple wordset for testing
const wordset = [
  "the",
  "be",
  "to",
  "of",
  "and",
  "a",
  "in",
  "that",
  "have",
  "i",
  "it",
  "for",
  "not",
  "on",
  "with",
  "he",
  "as",
  "you",
  "do",
  "at",
  "this",
  "but",
  "his",
  "by",
  "from",
  "they",
  "we",
  "say",
  "her",
  "she",
  "or",
  "an",
  "will",
  "my",
  "one",
  "all",
  "would",
  "there",
  "their",
  "what",
  "so",
  "up",
  "out",
  "if",
  "about",
  "who",
  "get",
  "which",
  "go",
  "me",
];

async function debugConstrainedGeneration(): Promise<void> {
  console.log("Loading model...");
  const modelId = "Xenova/distilgpt2";
  
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
  const generator = await pipeline("text-generation", modelId);
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const gen = generator as TextGenerationPipeline;

  // Get tokenizer
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
  const tokenizer = (gen as any).tokenizer;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
  const vocabSize = (gen as any).model?.config?.vocab_size ?? 50257;

  console.log(`Model loaded. Vocab size: ${vocabSize}`);
  console.log(`Wordset size: ${wordset.length} words`);

  // Create constrained processor
  const processor = new WordsetConstrainedLogitsProcessor(
    wordset,
    tokenizer,
    vocabSize,
    { debug: true }
  );

  // Helper to get valid tokens for current state
  function getValidTokens(): Array<{ id: number; text: string }> {
    const validTokenIds = processor.getValidTokenIds();

    return validTokenIds.map((tokenId) => {
      try {
        const decoded = tokenizer.decode([tokenId], {
          skip_special_tokens: false,
        });
        return { id: tokenId, text: decoded };
      } catch {
        return { id: tokenId, text: `[token ${tokenId}]` };
      }
    });
  }

  // Helper to get state info
  function getStateInfo(): string {
    const state = processor.getCurrentStateInfo();
    return `partialWord="${state.partialWord}", decodedText="${state.decodedText.substring(0, 30)}"`;
  }

  // Step-by-step generation
  let step = 0;
  const maxSteps = 20;
  const generatedTokens: bigint[] = [];

  processor.reset();

  console.log("=".repeat(80));
  console.log("Starting constrained generation (step by step)");
  console.log("=".repeat(80));

  while (step < maxSteps) {
    step++;
    console.log(`\n--- Step ${step} ---`);
    console.log(`State info: ${getStateInfo()}`);

    // Get valid tokens
    const validTokens = getValidTokens();
    console.log(`\nValid tokens (${validTokens.length} total):`);

    // Show first 20 valid tokens
    const sampleSize = Math.min(20, validTokens.length);
    for (let i = 0; i < sampleSize; i++) {
      const token = validTokens[i];
      if (token) {
        // Escape special characters for display
        const displayText = JSON.stringify(token.text);
        console.log(`  Token ${token.id}: ${displayText}`);
      }
    }
    if (validTokens.length > sampleSize) {
      console.log(`  ... and ${validTokens.length - sampleSize} more`);
    }

    if (validTokens.length === 0) {
      console.log("\n❌ ERROR: No valid tokens! Generation will fail.");
      break;
    }

    // Sample a token (using the processor to mask logits)
    const logits = new Float32Array(vocabSize);
    logits.fill(0); // Initialize with uniform logits

    // Pass the sequence of tokens generated so far
    // The processor will update its internal state based on these
    processor.process([generatedTokens], { data: logits, dims: [1, vocabSize] });

    // Find valid (non-masked) tokens
    const validLogits: Array<{ id: number; logit: number; text: string }> = [];
    for (let i = 0; i < vocabSize; i++) {
      const logit = logits[i];
      if (logit !== undefined && logit !== -Infinity && !isNaN(logit)) {
        try {
          const decoded = tokenizer.decode([i], { skip_special_tokens: false });
          validLogits.push({ id: i, logit, text: decoded });
        } catch {
          validLogits.push({ id: i, logit, text: `[token ${i}]` });
        }
      }
    }

    console.log(`\nAfter masking: ${validLogits.length} tokens with valid logits`);

    if (validLogits.length === 0) {
      console.log("\n❌ ERROR: No valid logits after masking!");
      break;
    }

    // Sample using temperature sampling
    const temperature = 1.0;
    const validLogitsArray = validLogits.map((t) => t.logit / temperature);
    const maxLogit = Math.max(...validLogitsArray);
    const expLogits = validLogitsArray.map((l) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((e) => e / sumExp);

    // Sample
    let random = Math.random();
    let sampledIdx = 0;
    for (let i = 0; i < probs.length; i++) {
      const prob = probs[i];
      if (prob === undefined) continue;
      random -= prob;
      if (random <= 0) {
        sampledIdx = i;
        break;
      }
    }

    const sampledToken = validLogits[sampledIdx];
    if (!sampledToken) {
      console.log("\n❌ ERROR: Failed to sample token!");
      break;
    }

    console.log(`\n🎲 Sampled token:`);
    console.log(`   ID: ${sampledToken.id}`);
    console.log(`   Text: ${JSON.stringify(sampledToken.text)}`);
    console.log(`   Logit: ${sampledToken.logit.toFixed(4)}`);
    const prob = probs[sampledIdx];
    if (prob !== undefined) {
      console.log(`   Probability: ${(prob * 100).toFixed(2)}%`);
    }

    // Add the sampled token to our sequence
    generatedTokens.push(BigInt(sampledToken.id));
    
    // Update state for the newly sampled token
    // process() already updated state for previous tokens, so we just need to update for this one
    processor.updateStateFromToken(sampledToken.id);
    
    // Update processedTokenCount to match the new length so process() doesn't double-process
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (processor as any).processedTokenCount = generatedTokens.length;

    // Check if we completed a word
    const stateInfo = processor.getCurrentStateInfo();
    const currentWords = processor.getGeneratedWords();
    if (stateInfo.partialWord === "" && currentWords.length > 0) {
      // We're at a word boundary, show the last completed word
      const lastWord = currentWords[currentWords.length - 1];
      if (lastWord) {
        console.log(`\n✅ Completed word: "${lastWord}"`);
      }
    }

    // Check if we should continue
    if (sampledToken.text.trim() === "" || sampledToken.text.includes("\n")) {
      console.log("\n⚠️  Generated space/newline - could continue or stop");
    }

    // Small delay for readability
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  console.log("\n" + "=".repeat(80));
  console.log("Generation complete");
  console.log("=".repeat(80));
  const finalWords = processor.getGeneratedWords();
  console.log(`\nGenerated words: ${JSON.stringify(finalWords)}`);
}

// Run if called directly (Node.js)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
if (typeof process !== "undefined" && (process as any).argv) {
  debugConstrainedGeneration()
    .then(() => {
      console.log("\n✅ Debug session complete");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      process.exit(0);
    })
    .catch((error) => {
      console.error("\n❌ Error:", error);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      process.exit(1);
    });
}

export { debugConstrainedGeneration };

