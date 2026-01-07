import { describe, it, expect } from "vitest";
import { WordsetDFA } from "../../../src/ts/test/llm/wordset-dfa";

// Mock tokenizer for testing
function createMockTokenizer(vocab: string[]): {
  decode: (
    tokens: bigint[] | number[],
    options?: { skip_special_tokens?: boolean }
  ) => string;
} {
  return {
    decode: (tokens: bigint[] | number[]) => {
      const tokenId = Number(tokens[0]);
      return vocab[tokenId] ?? "";
    },
  };
}

describe("WordsetDFA", () => {
  describe("constructor", () => {
    it("should create a DFA with initial words", () => {
      const dfa = new WordsetDFA(["hello", "world", "test"]);
      expect(dfa.size()).toBe(3);
      expect(dfa.getStateCount()).toBeGreaterThan(0);
    });

    it("should normalize words to lowercase for matching", () => {
      const dfa = new WordsetDFA(["Hello", "WORLD", "TeSt"]);
      expect(dfa.size()).toBe(3);
    });

    it("should filter words by min/max length", () => {
      const dfa = new WordsetDFA(["a", "ab", "abc", "abcdefghijklmnop"], {
        minWordLength: 2,
        maxWordLength: 5,
      });
      // "a" is too short, "abcdefghijklmnop" is too long
      expect(dfa.size()).toBe(2); // "ab" and "abc"
    });

    it("should handle empty wordset", () => {
      const dfa = new WordsetDFA([]);
      expect(dfa.size()).toBe(0);
    });
  });

  describe("getInitialState", () => {
    it("should return initial state (empty, at word boundary)", () => {
      const dfa = new WordsetDFA(["hello", "world"]);
      const initialState = dfa.getInitialState();
      expect(initialState.partialWord).toBe("");
      expect(initialState.atWordBoundary).toBe(true);
      expect(initialState.isCompleteWord).toBe(false);
    });
  });

  describe("transition", () => {
    it("should transition from initial state with a word-start token", () => {
      const dfa = new WordsetDFA(["hello", "world"]);
      const initialState = dfa.getInitialState();

      const result = dfa.transition(initialState, "hello");
      expect(result.isValid).toBe(true);
      expect(result.nextState).not.toBeNull();
      expect(result.nextState?.partialWord).toBe("hello");
      expect(result.nextState?.isCompleteWord).toBe(true);
      expect(result.nextState?.atWordBoundary).toBe(false);
      expect(result.completedWord).toBeNull(); // Not completed yet, just started
    });

    it("should complete a word when encountering a space", () => {
      const dfa = new WordsetDFA(["hello"]);
      const initialState = dfa.getInitialState();

      // First transition: "hello"
      const result1 = dfa.transition(initialState, "hello");
      expect(result1.isValid).toBe(true);
      expect(result1.nextState?.isCompleteWord).toBe(true);

      // Second transition: " " (space completes the word)
      const result2 = dfa.transition(result1.nextState!, " ");
      expect(result2.isValid).toBe(true);
      expect(result2.completedWord).toBe("hello");
      expect(result2.nextState?.atWordBoundary).toBe(true);
      expect(result2.nextState?.partialWord).toBe("");
    });

    it("should handle multi-character tokens", () => {
      const dfa = new WordsetDFA(["hello", "world"]);
      const initialState = dfa.getInitialState();

      // Token " hello" (space + word)
      const result = dfa.transition(initialState, " hello");
      expect(result.isValid).toBe(true);
      expect(result.completedWord).toBeNull(); // Word not completed yet
      expect(result.nextState?.partialWord).toBe("hello");
      expect(result.nextState?.isCompleteWord).toBe(true);
    });

    it("should handle partial word tokens", () => {
      const dfa = new WordsetDFA(["hello", "help"]);
      const initialState = dfa.getInitialState();

      // Token "he"
      const result1 = dfa.transition(initialState, "he");
      expect(result1.isValid).toBe(true);
      expect(result1.nextState?.partialWord).toBe("he");
      expect(result1.nextState?.isCompleteWord).toBe(false);

      // Token "llo" (continuation)
      const result2 = dfa.transition(result1.nextState!, "llo");
      expect(result2.isValid).toBe(true);
      expect(result2.nextState?.partialWord).toBe("hello");
      expect(result2.nextState?.isCompleteWord).toBe(true);
    });

    it("should reject invalid prefixes", () => {
      const dfa = new WordsetDFA(["hello"]);
      const initialState = dfa.getInitialState();

      const result = dfa.transition(initialState, "xyz");
      expect(result.isValid).toBe(false);
      expect(result.nextState).toBeNull();
    });

    it("should reject tokens that exceed max length", () => {
      const dfa = new WordsetDFA(["hello"], { maxWordLength: 3 });
      const initialState = dfa.getInitialState();

      const result = dfa.transition(initialState, "hello");
      expect(result.isValid).toBe(false);
    });

    it("should handle words that are prefixes of other words", () => {
      const dfa = new WordsetDFA(["the", "there", "their", "they"]);
      const initialState = dfa.getInitialState();

      // "the" is a complete word
      const result1 = dfa.transition(initialState, "the");
      expect(result1.isValid).toBe(true);
      expect(result1.nextState?.isCompleteWord).toBe(true);

      // Can continue to "there"
      const result2 = dfa.transition(result1.nextState!, "re");
      expect(result2.isValid).toBe(true);
      expect(result2.nextState?.partialWord).toBe("there");
      expect(result2.nextState?.isCompleteWord).toBe(true);
    });

    it("should handle case-insensitive matching", () => {
      const dfa = new WordsetDFA(["hello"]);
      const initialState = dfa.getInitialState();

      const result = dfa.transition(initialState, "HELLO");
      expect(result.isValid).toBe(true);
      expect(result.nextState?.partialWord).toBe("hello");
    });

    it("should ignore non-alphabetic characters in tokens", () => {
      const dfa = new WordsetDFA(["hello"]);
      const initialState = dfa.getInitialState();

      // Token with punctuation should still work if alphabetic part is valid
      const result = dfa.transition(initialState, "hello!");
      expect(result.isValid).toBe(true);
      expect(result.nextState?.partialWord).toBe("hello");
    });
  });

  describe("precomputeTokenTransitions", () => {
    it("should precompute transitions for all tokens", () => {
      const vocab = [" ", "hello", "world", "xyz"];
      const wordset = ["hello", "world"];
      const tokenizer = createMockTokenizer(vocab);

      const dfa = new WordsetDFA(wordset);
      const transitions = dfa.precomputeTokenTransitions(tokenizer, vocab.length);

      expect(transitions.size).toBe(dfa.getStateCount());

      // Check initial state transitions
      const initialState = dfa.getInitialState();
      const initialStateTransitions = transitions.get(initialState.id);
      expect(initialStateTransitions).toBeDefined();

      // Token 1 ("hello") should be valid from initial state
      expect(initialStateTransitions?.get(1)).not.toBeNull();

      // Token 2 ("world") should be valid from initial state
      expect(initialStateTransitions?.get(2)).not.toBeNull();

      // Token 3 ("xyz") should be invalid from initial state
      expect(initialStateTransitions?.get(3)).toBeNull();
    });

    it("should handle multi-token word generation", () => {
      // Vocab where words are split across tokens
      const vocab = [" ", "the", "re", "ir", "y", " hello"];
      const wordset = ["the", "there", "their", "they", "hello"];
      const tokenizer = createMockTokenizer(vocab);

      const dfa = new WordsetDFA(wordset);
      const transitions = dfa.precomputeTokenTransitions(tokenizer, vocab.length);

      const initialState = dfa.getInitialState();
      const initialStateTransitions = transitions.get(initialState.id)!;

      // "the" should be valid
      const theStateId = initialStateTransitions.get(1);
      expect(theStateId).toBeDefined();

      // From "the" state, "re" should be valid (forms "there")
      if (theStateId !== undefined && theStateId !== null) {
        const theStateTransitions = transitions.get(theStateId as number);
        expect(theStateTransitions?.get(2)).not.toBeNull(); // "re"
        expect(theStateTransitions?.get(3)).not.toBeNull(); // "ir" -> "their"
        expect(theStateTransitions?.get(4)).not.toBeNull(); // "y" -> "they"
      }
    });

    it("should handle space tokens correctly", () => {
      const vocab = [" ", "hello", "\n"];
      const wordset = ["hello"];
      const tokenizer = createMockTokenizer(vocab);

      const dfa = new WordsetDFA(wordset);
      const transitions = dfa.precomputeTokenTransitions(tokenizer, vocab.length);

      const initialState = dfa.getInitialState();
      const result1 = dfa.transition(initialState, "hello");
      expect(result1.isValid).toBe(true);

      // From "hello" state, space should complete the word
      const helloState = result1.nextState!;
      const helloStateTransitions = transitions.get(helloState.id);
      expect(helloStateTransitions?.get(0)).not.toBeNull(); // " " is valid
      expect(helloStateTransitions?.get(2)).not.toBeNull(); // "\n" is valid
    });
  });

  describe("real-world scenarios", () => {
    it("should work with English 200 wordset sample", () => {
      const words = [
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

      const dfa = new WordsetDFA(words);
      expect(dfa.size()).toBe(50);

      const initialState = dfa.getInitialState();

      // Test common words
      expect(dfa.transition(initialState, "the").isValid).toBe(true);
      expect(dfa.transition(initialState, "there").isValid).toBe(true);
      expect(dfa.transition(initialState, "their").isValid).toBe(true);
      expect(dfa.transition(initialState, "xyz").isValid).toBe(false);
    });

    it("should handle typing simulation with multi-token words", () => {
      const words = ["hello", "help", "helpful", "hero", "heroic"];
      const dfa = new WordsetDFA(words);
      const initialState = dfa.getInitialState();

      // Simulate typing "hel"
      const result1 = dfa.transition(initialState, "hel");
      expect(result1.isValid).toBe(true);
      expect(result1.nextState?.partialWord).toBe("hel");

      // Continue with "p" -> "help"
      const result2 = dfa.transition(result1.nextState!, "p");
      expect(result2.isValid).toBe(true);
      expect(result2.nextState?.partialWord).toBe("help");
      expect(result2.nextState?.isCompleteWord).toBe(true);

      // Continue with "ful" -> "helpful"
      const result3 = dfa.transition(result2.nextState!, "ful");
      expect(result3.isValid).toBe(true);
      expect(result3.nextState?.partialWord).toBe("helpful");
      expect(result3.nextState?.isCompleteWord).toBe(true);
    });

    it("should handle word boundaries correctly", () => {
      const words = ["hello", "world"];
      const dfa = new WordsetDFA(words);
      const initialState = dfa.getInitialState();

      // Generate "hello "
      const result1 = dfa.transition(initialState, "hello");
      expect(result1.isValid).toBe(true);
      expect(result1.completedWord).toBeNull();

      const result2 = dfa.transition(result1.nextState!, " ");
      expect(result2.isValid).toBe(true);
      expect(result2.completedWord).toBe("hello");
      expect(result2.nextState?.atWordBoundary).toBe(true);

      // Now can start new word
      const result3 = dfa.transition(result2.nextState!, "world");
      expect(result3.isValid).toBe(true);
      expect(result3.nextState?.partialWord).toBe("world");
      expect(result3.nextState?.isCompleteWord).toBe(true);
    });
  });

  describe("edge cases", () => {
    it("should handle empty tokens", () => {
      const dfa = new WordsetDFA(["hello"]);
      const initialState = dfa.getInitialState();

      const result = dfa.transition(initialState, "");
      // Empty token might be invalid or might be a no-op
      // The behavior depends on implementation, but should not crash
      expect(result).toBeDefined();
    });

    it("should handle tokens with only spaces", () => {
      const dfa = new WordsetDFA(["hello"]);
      const initialState = dfa.getInitialState();

      const result = dfa.transition(initialState, "   ");
      expect(result.isValid).toBe(true);
      expect(result.nextState?.atWordBoundary).toBe(true);
    });

    it("should handle very long tokens", () => {
      const dfa = new WordsetDFA(["hello"], { maxWordLength: 10 });
      const initialState = dfa.getInitialState();

      const result = dfa.transition(initialState, "helloworldtoolong");
      expect(result.isValid).toBe(false);
    });

    it("should handle words with special characters", () => {
      // DFA should extract alphabetic characters
      const dfa = new WordsetDFA(["hello", "world"]);
      const initialState = dfa.getInitialState();

      // Token with special chars should still match if alphabetic part is valid
      const result = dfa.transition(initialState, "hello123");
      expect(result.isValid).toBe(true);
      expect(result.nextState?.partialWord).toBe("hello");
    });
  });
});





