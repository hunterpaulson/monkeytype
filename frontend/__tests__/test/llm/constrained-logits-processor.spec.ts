import { describe, it, expect, beforeEach, vi } from "vitest";
import { WordsetConstrainedLogitsProcessor } from "../../../src/ts/test/llm/constrained-logits-processor";

// Mock tokenizer that simulates GPT-2 style BPE tokens
function createMockTokenizer(vocab: string[]): {
  decode: (
    tokens: bigint[] | number[],
    options?: { skip_special_tokens?: boolean }
  ) => string;
  vocab_size: number;
} {
  return {
    decode: (tokens: bigint[] | number[]) => {
      const tokenId = Number(tokens[0]);
      return vocab[tokenId] ?? "";
    },
    vocab_size: vocab.length,
  };
}

// Create a tensor-like object for testing
function createMockLogits(size: number, initialValue: number = 0): {
  data: Float32Array;
  dims: number[];
} {
  const data = new Float32Array(size);
  data.fill(initialValue);
  return { data, dims: [1, size] };
}

describe("WordsetConstrainedLogitsProcessor", () => {
  // Simple vocab for testing
  // Token 0: " " (space)
  // Token 1: "the"
  // Token 2: "re" (continuation)
  // Token 3: "ir" (continuation)
  // Token 4: "y" (continuation)
  // Token 5: " hello"
  // Token 6: "xyz" (invalid continuation)
  // Token 7: "\n" (newline)
  const vocab = [" ", "the", "re", "ir", "y", " hello", "xyz", "\n"];
  const wordset = ["the", "there", "their", "they", "hello"];

  let processor: WordsetConstrainedLogitsProcessor;
  let tokenizer: ReturnType<typeof createMockTokenizer>;

  beforeEach(() => {
    tokenizer = createMockTokenizer(vocab);
    // Suppress console output during tests
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    processor = new WordsetConstrainedLogitsProcessor(
      wordset,
      tokenizer,
      vocab.length
    );
  });

  describe("constructor", () => {
    it("should initialize with wordset", () => {
      expect(processor).toBeDefined();
    });
  });

  describe("reset", () => {
    it("should clear internal state", () => {
      // Process some tokens to build up state
      const logits = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1)]], logits); // "the"

      processor.reset();

      expect(processor.getGeneratedWords()).toEqual([]);
    });
  });

  describe("process", () => {
    it("should mask invalid continuations", () => {
      // Start fresh - no partial word
      const logits = createMockLogits(vocab.length, 1.0);

      // Process with empty input (start of generation)
      processor.process([[]], logits);

      // Check that some tokens are masked (set to -Infinity)
      // Token 6 ("xyz") should be masked as "xyz" is not a valid prefix
      expect(logits.data[6]).toBe(-Infinity);
    });

    it("should allow space tokens", () => {
      const logits = createMockLogits(vocab.length, 1.0);
      processor.process([[]], logits);

      // Space and newline should be allowed (for word boundaries)
      expect(logits.data[0]).not.toBe(-Infinity); // " "
      expect(logits.data[7]).not.toBe(-Infinity); // "\n"
    });

    it("should track generated words", () => {
      // Generate "the " (word followed by space)
      const logits1 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1)]], logits1); // "the"

      const logits2 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(0)]], logits2); // "the" + " "

      expect(processor.getGeneratedWords()).toEqual(["the"]);
    });

    it("should allow valid continuations", () => {
      // At the start (no tokens yet), any word-start token should be valid
      const logits1 = createMockLogits(vocab.length, 1.0);
      processor.process([[]], logits1); // Empty input - start of generation

      // Token 1 ("the") should be allowed since "the" is a valid word/prefix
      expect(logits1.data[1]).not.toBe(-Infinity);

      // Token 5 (" hello") should be allowed since it starts with space + valid word
      expect(logits1.data[5]).not.toBe(-Infinity);

      // Token 6 ("xyz") should be masked since "xyz" is not in wordset
      expect(logits1.data[6]).toBe(-Infinity);
    });

    it("should handle multi-token word generation", () => {
      // Test that we can generate "there" from "the" + "re"
      const logits1 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1)]], logits1); // "the"

      // After "the", "re" should be a valid continuation (forms "there")
      const logits2 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(2)]], logits2); // "the" + "re"

      // "re" should be allowed
      expect(logits2.data[2]).not.toBe(-Infinity);
    });
  });

  describe("getProcessor", () => {
    it("should return a callable function", () => {
      const processorFn = processor.getProcessor();
      expect(typeof processorFn).toBe("function");

      const logits = createMockLogits(vocab.length, 1.0);
      const result = processorFn([[]], logits);
      expect(result).toBe(logits);
    });
  });
});
