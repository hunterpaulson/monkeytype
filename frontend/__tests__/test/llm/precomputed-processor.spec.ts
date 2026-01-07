import { describe, it, expect, beforeEach, vi } from "vitest";
import { PrecomputedConstrainedProcessor } from "../../../src/ts/test/llm/precomputed-constrained-processor";
import { BenchmarkContext } from "../../../src/ts/test/llm/benchmark";

/**
 * Create a mock tokenizer that maps token IDs to strings
 */
function createMockTokenizer(vocab: Record<number, string>): {
  decode: (
    tokens: Array<number | bigint>,
    options?: { skip_special_tokens?: boolean }
  ) => string;
} {
  return {
    decode: (tokens: Array<number | bigint>) => {
      return tokens
        .map((t) => {
          const id = typeof t === "bigint" ? Number(t) : t;
          return vocab[id] ?? "";
        })
        .join("");
    },
  };
}

/**
 * Create a mock logits tensor
 */
function createMockLogits(
  size: number,
  initialValue: number = 0
): { data: Float32Array; dims: number[] } {
  const data = new Float32Array(size);
  data.fill(initialValue);
  return { data, dims: [1, size] };
}

describe("PrecomputedConstrainedProcessor", () => {
  beforeEach(() => {
    // Suppress console output during tests
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  describe("initialization", () => {
    it("should initialize with a simple wordset", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: "hello",
        2: "world",
        3: " hello",
        4: " world",
        5: "xyz",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello", "world"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      expect(processor).toBeDefined();
      const stats = processor.getStats();
      expect(stats.stateCount).toBeGreaterThan(0);
    });

    it("should track initialization timing with benchmark context", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " hello",
        2: " world",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello", "world"];
      const benchmark = new BenchmarkContext();

      new PrecomputedConstrainedProcessor(wordset, tokenizer, 3, { benchmark });

      const result = benchmark.getResult();
      expect(result.initTime.totalMs).toBeGreaterThan(0);
      expect(result.initTime.tokenCacheMs).toBeGreaterThanOrEqual(0);
      expect(result.initTime.stateMachineBuildMs).toBeGreaterThanOrEqual(0);
    });

    it("should report progress during initialization", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " hello",
        2: " world",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello", "world"];
      const progressCalls: Array<{ phase: string; percent: number }> = [];

      new PrecomputedConstrainedProcessor(wordset, tokenizer, 3, {
        onProgress: (progress) => {
          progressCalls.push({ phase: progress.phase, percent: progress.percent });
        },
      });

      // Should have caching, building, and ready phases
      expect(progressCalls.some((p) => p.phase === "caching")).toBe(true);
      expect(progressCalls.some((p) => p.phase === "building")).toBe(true);
      expect(progressCalls.some((p) => p.phase === "ready")).toBe(true);
    });
  });

  describe("state machine construction", () => {
    it("should have initial state with valid word-starting tokens", () => {
      // Vocabulary with space-prefixed words
      const vocab: Record<number, string> = {
        0: " ",
        1: " h",
        2: " he",
        3: " hel",
        4: " hell",
        5: " hello",
        6: " w",
        7: " wo",
        8: " wor",
        9: " worl",
        10: " world",
        11: "xyz",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello", "world"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // From initial state, we should be able to start "hello" or "world"
      const validTokens = processor.getValidTokenIds();

      // Tokens that start valid prefixes should be valid
      // " h", " he", " hel", " hell", " hello" all start "hello"
      // " w", " wo", " wor", " worl", " world" all start "world"
      expect(validTokens.length).toBeGreaterThan(0);

      // "xyz" should NOT be valid
      expect(validTokens).not.toContain(11);
    });

    it("should create multiple states for different prefixes", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " h",
        2: "e",
        3: "l",
        4: "lo",
        5: " hello",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      const stats = processor.getStats();
      // Should have multiple states: initial, " h", " he", " hel", " hell", " hello", etc.
      expect(stats.stateCount).toBeGreaterThan(1);
    });
  });

  describe("logits masking", () => {
    it("should mask invalid tokens from initial state", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " hello",
        2: " world",
        3: "xyz",
        4: "!@#",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello", "world"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      const logits = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[]], logits);

      // " hello" and " world" should be valid (complete words with space)
      expect(logits.data[1]).not.toBe(-Infinity); // " hello"
      expect(logits.data[2]).not.toBe(-Infinity); // " world"

      // "xyz" and "!@#" should be invalid
      expect(logits.data[3]).toBe(-Infinity); // "xyz"
      expect(logits.data[4]).toBe(-Infinity); // "!@#"
    });

    it("should allow continuations after partial word", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " h",
        2: "e",
        3: "l",
        4: "l",
        5: "o",
        6: " hello",
        7: "x",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // Simulate generating " h" (token 1)
      const logits1 = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[BigInt(1)]], logits1);

      // After " h", we should be able to continue with "e"
      const logits2 = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[BigInt(1), BigInt(2)]], logits2); // " h" + "e" = " he"

      // "l" should be valid to continue " hel"
      expect(logits2.data[3]).not.toBe(-Infinity);

      // "x" should be invalid
      expect(logits2.data[7]).toBe(-Infinity);
    });

    it("should reset to initial state after completing a word", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " hello",
        2: " world",
        3: "xyz",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello", "world"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // Generate " hello" (complete word)
      const logits1 = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[BigInt(1)]], logits1); // " hello"

      // After completing " hello", we should be back at word boundary
      const stateInfo1 = processor.getCurrentStateInfo();
      expect(stateInfo1.prefix).toBe(""); // " hello" is complete

      // Now we can start " world"
      const logits2 = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[BigInt(1), BigInt(2)]], logits2); // " hello" + " world"

      // After " hello world", both words are complete
      const stateInfo2 = processor.getCurrentStateInfo();
      // The decoded text is " hello world" - both words complete, back to empty prefix
      expect(stateInfo2.prefix).toBe("");
    });
  });

  describe("word extraction", () => {
    it("should track generated words", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " hello",
        2: " world",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello", "world"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // Generate " hello world"
      processor.process([[BigInt(1), BigInt(2)]], createMockLogits(3, 1.0));

      const words = processor.getGeneratedWords();
      expect(words).toContain("hello");
      expect(words).toContain("world");
    });

    it("should reset state and words", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " hello",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // Generate a word
      processor.process([[BigInt(1)]], createMockLogits(2, 1.0));
      expect(processor.getGeneratedWords().length).toBeGreaterThan(0);

      // Reset
      processor.reset();

      // Should be back to initial state
      const stateInfo = processor.getCurrentStateInfo();
      expect(stateInfo.stateId).toBe(0);
      expect(stateInfo.prefix).toBe("");
    });
  });

  describe("edge cases", () => {
    it("should handle empty input", () => {
      const vocab: Record<number, string> = {
        0: " hello",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // Empty input should work
      const logits = createMockLogits(1, 1.0);
      processor.process([[]], logits);

      expect(logits.data[0]).not.toBe(-Infinity);
    });

    it("should handle single-character words", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " a",
        2: " I",
        3: "x",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["a", "I"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      const logits = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[]], logits);

      // " a" and " I" should be valid
      expect(logits.data[1]).not.toBe(-Infinity);
      expect(logits.data[2]).not.toBe(-Infinity);

      // "x" should be invalid (no space prefix)
      expect(logits.data[3]).toBe(-Infinity);
    });

    it("should handle overlapping word prefixes", () => {
      // "the", "there", "their", "they" all share "the" prefix
      const vocab: Record<number, string> = {
        0: " ",
        1: " the",
        2: "re",
        3: "ir",
        4: "y",
        5: " there",
        6: " their",
        7: " they",
        8: "x",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["the", "there", "their", "they"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // From initial state
      const logits1 = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[]], logits1);

      // " the" should be valid (it's both a complete word and prefix)
      expect(logits1.data[1]).not.toBe(-Infinity);

      // After " the", we can continue with "re", "ir", "y" or start new word
      const logits2 = createMockLogits(Object.keys(vocab).length, 1.0);
      processor.process([[BigInt(1)]], logits2); // " the"

      // Continuations should be valid
      expect(logits2.data[2]).not.toBe(-Infinity); // "re" -> "there"
      expect(logits2.data[3]).not.toBe(-Infinity); // "ir" -> "their"
      expect(logits2.data[4]).not.toBe(-Infinity); // "y" -> "they"

      // Space should also be valid (complete "the" and start new word)
      expect(logits2.data[0]).not.toBe(-Infinity);
    });
  });

  describe("processor function", () => {
    it("should return a callable processor function", () => {
      const vocab: Record<number, string> = {
        0: " hello",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      const processorFn = processor.getProcessor();
      expect(typeof processorFn).toBe("function");

      const logits = createMockLogits(1, 1.0);
      const result = processorFn([[]], logits);
      expect(result).toBe(logits);
    });
  });

  describe("state info", () => {
    it("should return current state info", () => {
      const vocab: Record<number, string> = {
        0: " ",
        1: " h",
        2: "ello",
      };
      const tokenizer = createMockTokenizer(vocab);
      const wordset = ["hello"];

      const processor = new PrecomputedConstrainedProcessor(
        wordset,
        tokenizer,
        Object.keys(vocab).length
      );

      // Initial state
      let info = processor.getCurrentStateInfo();
      expect(info.stateId).toBe(0);
      expect(info.prefix).toBe("");

      // After " h"
      processor.process([[BigInt(1)]], createMockLogits(3, 1.0));
      info = processor.getCurrentStateInfo();
      expect(info.prefix).toBe(" h");
    });
  });
});

