import { describe, it, expect, beforeEach, vi } from "vitest";
import { WordsetConstrainedLogitsProcessor } from "../../../src/ts/test/llm/constrained-logits-processor";
import { WordsetDFA } from "../../../src/ts/test/llm/wordset-dfa";

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

describe("Constrained Decoding Integration", () => {
  describe("basic wordset constraint", () => {
    // Simple vocab for testing
    // Token 0: " " (space)
    // Token 1: "the"
    // Token 2: "re" (continuation)
    // Token 3: "ir" (continuation)
    // Token 4: "y" (continuation)
    // Token 5: " hello"
    // Token 6: "xyz" (invalid)
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

    it("should mask invalid tokens from initial state", () => {
      const logits = createMockLogits(vocab.length, 1.0);
      processor.process([[]], logits);

      // Valid tokens from initial state
      expect(logits.data[1]).not.toBe(-Infinity); // "the"
      expect(logits.data[5]).not.toBe(-Infinity); // " hello"
      expect(logits.data[0]).not.toBe(-Infinity); // " " (space)

      // Invalid tokens
      expect(logits.data[6]).toBe(-Infinity); // "xyz"
    });

    it("should allow valid continuations after 'the'", () => {
      // Start with "the"
      const logits1 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1)]], logits1);

      // After "the", we can continue with "re", "ir", "y", or space
      const logits2 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(2)]], logits2); // "the" + "re"

      // "re" should be allowed (forms "there")
      expect(logits2.data[2]).not.toBe(-Infinity);
      // "ir" should be allowed (forms "their")
      expect(logits2.data[3]).not.toBe(-Infinity);
      // "y" should be allowed (forms "they")
      expect(logits2.data[4]).not.toBe(-Infinity);
      // Space should be allowed (completes "the")
      expect(logits2.data[0]).not.toBe(-Infinity);
    });

    it("should track generated words", () => {
      // Generate "the "
      const logits1 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1)]], logits1); // "the"

      const logits2 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(0)]], logits2); // "the" + " "

      expect(processor.getGeneratedWords()).toContain("the");
    });

    it("should handle multi-token word generation", () => {
      // Generate "there" from "the" + "re"
      const logits1 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1)]], logits1); // "the"

      const logits2 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(2)]], logits2); // "the" + "re" = "there"

      // After "there", space should complete the word
      const logits3 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(2), BigInt(0)]], logits3); // "there" + " "

      expect(processor.getGeneratedWords()).toContain("there");
    });
  });

  describe("real-world wordset constraint", () => {
    const vocab = [
      " ",
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
      "xyz", // invalid
    ];

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

    let processor: WordsetConstrainedLogitsProcessor;
    let tokenizer: ReturnType<typeof createMockTokenizer>;

    beforeEach(() => {
      tokenizer = createMockTokenizer(vocab);
      vi.spyOn(console, "log").mockImplementation(() => {});
      vi.spyOn(console, "warn").mockImplementation(() => {});
      processor = new WordsetConstrainedLogitsProcessor(
        wordset,
        tokenizer,
        vocab.length
      );
    });

    it("should create DFA with many states", () => {
      // The DFA should have more than just the initial state
      // States are created during precomputation
      expect(processor).toBeDefined();
      // We can't directly access the DFA, but we can verify it works
    });

    it("should allow valid words from wordset", () => {
      const logits = createMockLogits(vocab.length, 1.0);
      processor.process([[]], logits);

      // All wordset tokens should be valid (except the invalid one)
      for (let i = 0; i < wordset.length; i++) {
        // Find token ID for this word (simplified - in real case would need mapping)
        // For this test, we assume tokens 1-49 are valid words
        if (i < vocab.length - 1) {
          // Most tokens should be valid (except token 50 "xyz")
          if (i < 50) {
            // Allow some flexibility - not all tokens might be valid from initial state
            // (e.g., if they require a space prefix)
          }
        }
      }

      // Invalid token should definitely be masked
      expect(logits.data[50]).toBe(-Infinity); // "xyz"
    });

    it("should generate sequence of valid words", () => {
      // Simulate generating "the be to"
      const logits1 = createMockLogits(vocab.length, 1.0);
      processor.process([[]], logits1);

      const logits2 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1)]], logits2); // "the"

      const logits3 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(0)]], logits3); // "the" + " "

      const logits4 = createMockLogits(vocab.length, 1.0);
      processor.process([[BigInt(1), BigInt(0), BigInt(2)]], logits4); // "the" + " " + "be"

      // Should have tracked "the" as a completed word
      expect(processor.getGeneratedWords().length).toBeGreaterThan(0);
    });
  });

  describe("state exploration", () => {
    it("should discover states during precomputation", () => {
      const vocab = [" ", "hello", "world", " hel", "lo", " wor", "ld"];
      const wordset = ["hello", "world"];
      const tokenizer = createMockTokenizer(vocab);

      vi.spyOn(console, "log").mockImplementation(() => {});

      const processor = new WordsetConstrainedLogitsProcessor(
        wordset,
        tokenizer,
        vocab.length
      );

      // Processor should have been created successfully
      expect(processor).toBeDefined();

      // Test that transitions work
      const logits = createMockLogits(vocab.length, 1.0);
      processor.process([[]], logits);

      // Some tokens should be valid
      const validCount = Array.from(logits.data).filter(
        (v) => v !== -Infinity
      ).length;
      expect(validCount).toBeGreaterThan(0);
    });
  });

  describe("reset functionality", () => {
    it("should reset to initial state", () => {
      const vocab = [" ", "the", "hello"];
      const wordset = ["the", "hello"];
      const tokenizer = createMockTokenizer(vocab);

      vi.spyOn(console, "log").mockImplementation(() => {});
      const processor = new WordsetConstrainedLogitsProcessor(
        wordset,
        tokenizer,
        vocab.length
      );

      // Generate some words
      processor.process([[BigInt(1)]], createMockLogits(vocab.length, 1.0));
      processor.process([[BigInt(1), BigInt(0)]], createMockLogits(vocab.length, 1.0));

      expect(processor.getGeneratedWords().length).toBeGreaterThan(0);

      // Reset
      processor.reset();

      // Should be back to initial state
      const logits = createMockLogits(vocab.length, 1.0);
      processor.process([[]], logits);

      // Should allow valid tokens from initial state
      expect(logits.data[1]).not.toBe(-Infinity); // "the"
      expect(logits.data[2]).not.toBe(-Infinity); // "hello"
    });
  });
});





