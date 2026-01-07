/**
 * Constrained Logits Processor for Wordset-Based Generation
 *
 * Simplified brute-force approach: O(vocab_size) check at each step
 * to validate if a token is a valid continuation of the current sequence.
 *
 * Logic:
 * - Track current decoded text and partial word being built
 * - For each token in vocab, decode it and check if it's a valid continuation:
 *   - If at word boundary: token must start a word prefix OR be a space
 *   - If building a word: token must continue the partial word OR complete it with space
 */

// Type definitions matching transformers.js
type Tensor = {
  data: Float32Array;
  dims: number[];
};

type TokenizerDecodeFunction = (
  tokens: bigint[] | number[],
  options?: { skip_special_tokens?: boolean }
) => string;

type TokenizerLike = {
  decode: TokenizerDecodeFunction;
  vocab_size?: number;
};

/**
 * Configuration for the constrained logits processor
 */
export type ConstrainedProcessorConfig = {
  /** Minimum word length to allow */
  minWordLength: number;
  /** Maximum word length to allow */
  maxWordLength: number;
  /** Whether to allow the model to generate punctuation */
  allowPunctuation: boolean;
  /** Debug mode - log decisions */
  debug: boolean;
};

const DEFAULT_CONFIG: ConstrainedProcessorConfig = {
  minWordLength: 1,
  maxWordLength: 15,
  allowPunctuation: false,
  debug: false,
};

/**
 * Logits processor that constrains generation to words from a wordset.
 * Uses brute-force O(vocab_size) validation at each step.
 */
export class WordsetConstrainedLogitsProcessor {
  private wordset: Set<string>;
  private wordPrefixes: Set<string>;
  private tokenizer: TokenizerLike;
  private vocabSize: number;
  private config: ConstrainedProcessorConfig;

  // State tracking for current generation
  private decodedText: string = "";
  private partialWord: string = "";
  private processedTokenCount: number = 0;

  // Cache token strings to avoid repeated decoding
  private tokenStringCache: Map<number, string> = new Map();

  constructor(
    words: string[],
    tokenizer: TokenizerLike,
    vocabSize: number,
    config: Partial<ConstrainedProcessorConfig> = {}
  ) {
    this.tokenizer = tokenizer;
    this.vocabSize = vocabSize;
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Build wordset and prefix set
    this.wordset = new Set();
    this.wordPrefixes = new Set();

    for (const word of words) {
      const normalized = word.toLowerCase().trim();
      const wordChars = normalized.replace(/[^a-z]/g, "");

      if (wordChars.length === 0) continue;
      if (wordChars.length < this.config.minWordLength) continue;
      if (wordChars.length > this.config.maxWordLength) continue;

      this.wordset.add(wordChars);

      // Add all prefixes
      for (let i = 1; i <= wordChars.length; i++) {
        this.wordPrefixes.add(wordChars.substring(0, i));
      }
    }

    console.log(
      `[LLM Constrained] Initialized with ${this.wordset.size} words, ` +
        `${this.wordPrefixes.size} prefixes`
    );

    // Pre-cache all token strings
    this.precomputeTokenStrings();
  }

  /**
   * Pre-compute token strings to avoid repeated decoding
   */
  private precomputeTokenStrings(): void {
    console.log(`[LLM Constrained] Pre-computing token strings...`);
    for (let tokenId = 0; tokenId < this.vocabSize; tokenId++) {
      try {
        const decoded = this.tokenizer.decode([tokenId], {
          skip_special_tokens: false,
        });
        this.tokenStringCache.set(tokenId, decoded);
      } catch {
        this.tokenStringCache.set(tokenId, "");
      }
    }
    console.log(`[LLM Constrained] Cached ${this.tokenStringCache.size} token strings`);
  }

  /**
   * Extract alphabetic characters from a string (for word matching)
   */
  private extractWordChars(str: string): string {
    return str
      .toLowerCase()
      .replace(/[^a-z]/g, "");
  }

  /**
   * Check if a character is a word boundary (space, newline, tab)
   */
  private isWordBoundary(char: string): boolean {
    return char === " " || char === "\n" || char === "\t";
  }

  /**
   * Check if a token is a valid continuation of the current sequence
   */
  private isValidToken(tokenString: string): boolean {
    // If we're at a word boundary (partialWord is empty)
    if (this.partialWord === "") {
      // Token must either:
      // 1. Start a word prefix (has alphabetic chars that form a valid prefix)
      // 2. Be pure word boundaries (spaces/newlines)
      
      let hasAlphabetic = false;
      let hasWordBoundary = false;
      let hasOtherChars = false;
      let newPartial = "";

      for (const char of tokenString) {
        if (this.isWordBoundary(char)) {
          hasWordBoundary = true;
          // If we hit a boundary, check if we had a valid word before it
          if (newPartial.length > 0) {
            if (!this.wordset.has(newPartial)) {
              return false; // Incomplete word before boundary
            }
            // Word completed, reset
            newPartial = "";
          }
        } else if (/[a-z]/i.test(char)) {
          hasAlphabetic = true;
          newPartial += char.toLowerCase();
          // Check if this prefix is valid
          if (!this.wordPrefixes.has(newPartial)) {
            return false; // Invalid prefix
          }
        } else {
          hasOtherChars = true;
        }
      }

      // If we saw any non-boundary, non-alphabetic characters, reject the token.
      // This prevents strange unicode (like "μ"), digits, punctuation, etc.
      if (hasOtherChars) {
        return false;
      }

      // After processing token:
      // - If we have alphabetic chars, newPartial must be a valid prefix
      // - If only word boundaries, that's fine
      // - If word boundaries + other non-alphabetic chars, invalid
      if (hasAlphabetic) {
        return newPartial.length === 0 || this.wordPrefixes.has(newPartial);
      } else if (hasWordBoundary) {
        return !hasOtherChars; // Only valid if pure word boundaries
      } else {
        return false; // No alphabetic, no boundary = invalid
      }
    } else {
      // We're building a word - token must continue it
      let currentPartial = this.partialWord;
      let hasWordBoundary = false;
      let hasOtherChars = false;

      for (const char of tokenString) {
        if (this.isWordBoundary(char)) {
          hasWordBoundary = true;
          // Check if current partial is a complete word
          if (!this.wordset.has(currentPartial)) {
            return false; // Incomplete word before boundary
          }
          // Word completed, reset
          currentPartial = "";
        } else if (/[a-z]/i.test(char)) {
          currentPartial += char.toLowerCase();
          // Check if this continues to be a valid prefix
          if (!this.wordPrefixes.has(currentPartial)) {
            return false; // Invalid continuation
          }
          // Check max length
          if (currentPartial.length > this.config.maxWordLength) {
            return false;
          }
        } else {
          hasOtherChars = true;
        }
      }

      // If we saw any non-boundary, non-alphabetic characters, reject the token.
      if (hasOtherChars) {
        return false;
      }

      // After processing:
      // - If we hit a boundary, currentPartial should be empty (word completed)
      // - If we didn't hit a boundary, currentPartial should still be a valid prefix
      if (hasWordBoundary) {
        return currentPartial === "" && !hasOtherChars;
      } else {
        return this.wordPrefixes.has(currentPartial) && !hasOtherChars;
      }
    }
  }

  /**
   * Reset state for a new generation.
   */
  public reset(): void {
    this.decodedText = "";
    this.partialWord = "";
    this.processedTokenCount = 0;
  }

  /**
   * Get the list of words generated in the current session.
   * Extracts words from decodedText at the end (not during token processing).
   */
  public getGeneratedWords(): string[] {
    // Extract words from the full decoded text
    // Split on word boundaries and extract alphabetic sequences
    const words: string[] = [];
    let currentWord = "";
    
    for (const char of this.decodedText) {
      if (this.isWordBoundary(char)) {
        if (currentWord.length > 0) {
          const normalized = currentWord.toLowerCase().replace(/[^a-z]/g, "");
          if (normalized.length > 0 && this.wordset.has(normalized)) {
            words.push(normalized);
          }
          currentWord = "";
        }
      } else if (/[a-z]/i.test(char)) {
        currentWord += char;
      }
    }
    
    // Handle last word if text doesn't end with boundary
    if (currentWord.length > 0) {
      const normalized = currentWord.toLowerCase().replace(/[^a-z]/g, "");
      if (normalized.length > 0 && this.wordset.has(normalized)) {
        words.push(normalized);
      }
    }
    
    return words;
  }

  /**
   * Get current state info (for debugging)
   */
  public getCurrentStateInfo(): { decodedText: string; partialWord: string } {
    return {
      decodedText: this.decodedText,
      partialWord: this.partialWord,
    };
  }

  /**
   * Get valid token IDs for the current state (for debugging)
   */
  public getValidTokenIds(): number[] {
    const valid: number[] = [];
    for (let tokenId = 0; tokenId < this.vocabSize; tokenId++) {
      const tokenString = this.tokenStringCache.get(tokenId) ?? "";
      if (this.isValidToken(tokenString)) {
        valid.push(tokenId);
      }
    }
    return valid;
  }

  /**
   * Update state based on a generated token
   * Public for debugging/testing purposes
   * 
   * NOTE: We do NOT extract completed words here - that happens at the end
   * from the full decodedText. We only track partialWord for validation purposes.
   */
  public updateStateFromToken(tokenId: number): void {
    const tokenString = this.tokenStringCache.get(tokenId) ?? "";
    this.decodedText += tokenString;

    // Process the token to update partialWord (for validation only)
    // We don't extract completed words here - that happens at the end
    let newPartial = this.partialWord;

    for (const char of tokenString) {
      if (this.isWordBoundary(char)) {
        // Word boundary encountered - reset partial word
        newPartial = "";
      } else if (/[a-z]/i.test(char)) {
        // Alphabetic character - continue building word (always lowercase)
        newPartial += char.toLowerCase();
      }
    }

    this.partialWord = newPartial;
  }

  /**
   * The main logits processing function.
   * Called by transformers.js at each generation step.
   */
  public process(input_ids: bigint[][], logits: Tensor): Tensor {
    // Process each batch item
    for (let batchIdx = 0; batchIdx < input_ids.length; batchIdx++) {
      const batchLogits = logits.data.subarray(
        batchIdx * this.vocabSize,
        (batchIdx + 1) * this.vocabSize
      );

      this.processLogits(input_ids[batchIdx] ?? [], batchLogits);
    }

    return logits;
  }

  /**
   * Process logits for a single batch item.
   */
  private processLogits(inputIds: bigint[], logits: Float32Array): void {
    // Process only new tokens (ones we haven't processed yet)
    for (let i = this.processedTokenCount; i < inputIds.length; i++) {
      const tokenId = Number(inputIds[i]);
      this.updateStateFromToken(tokenId);
    }

    // Update processed count
    this.processedTokenCount = inputIds.length;

    // Brute-force validation: check each token
    let validCount = 0;
    const validTokenIds: number[] = [];

    for (let tokenId = 0; tokenId < this.vocabSize; tokenId++) {
      const tokenString = this.tokenStringCache.get(tokenId) ?? "";
      if (this.isValidToken(tokenString)) {
        validCount++;
        if (validCount <= 10) {
          validTokenIds.push(tokenId);
        }
      } else {
        // Invalid token - mask it
        logits[tokenId] = -Infinity;
      }
    }

    // Log state info for debugging
    if (this.config.debug || this.processedTokenCount === 0) {
      if (this.processedTokenCount === 0) {
        console.log(
          `[LLM Constrained] Initial state: ${validCount} valid tokens available`
        );
        // Decode a few to see what they are
        if (validTokenIds.length > 0) {
          try {
            const sampleTokens = validTokenIds.slice(0, 5).map((id) => {
              const tokenStr = this.tokenStringCache.get(id) ?? "";
              return JSON.stringify(tokenStr);
            });
            console.log(
              `[LLM Constrained] Sample valid tokens: ${sampleTokens.join(", ")}`
            );
          } catch {
            // Ignore errors
          }
        }
      } else if (this.config.debug && validCount < 100) {
        console.log(
          `[LLM Constrained] State: partialWord="${this.partialWord}", ` +
            `valid tokens: ${validCount}`
        );
      }
    }

    // Safety: if we've masked everything, log error
    if (validCount === 0) {
      console.error(
        `[LLM Constrained] CRITICAL: No valid tokens! ` +
          `Current state: partialWord="${this.partialWord}", ` +
          `decodedText="${this.decodedText.substring(0, 50)}"`
      );
    }
  }

  /**
   * Create a callable that can be passed to transformers.js logits_processor.
   */
  public getProcessor(): (input_ids: bigint[][], logits: Tensor) => Tensor {
    return (input_ids: bigint[][], logits: Tensor) => {
      return this.process(input_ids, logits);
    };
  }

  /**
   * Get DFA instance (for compatibility with debug script)
   * Returns null since we're not using DFA anymore
   */
  public getDFA(): null {
    return null;
  }

  /**
   * Get current state ID (for compatibility with debug script)
   * Returns 0 as a placeholder
   */
  public getCurrentStateId(): number {
    return 0;
  }
}
