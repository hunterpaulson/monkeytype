/**
 * Precomputed Constrained Logits Processor
 *
 * Achieves O(1) per-token constraint checking by precomputing all valid
 * transitions at initialization time.
 *
 * Architecture:
 * - Build a trie from wordset (words stored as " word" with leading space)
 * - States = unique prefixes encountered during BFS exploration
 * - For each state, precompute: valid token IDs and next state for each valid token
 * - During generation: simple Map lookups instead of expensive validation
 *
 * Optimizations:
 * - Pre-filter tokens that can never be valid (contain invalid characters)
 * - Group tokens by type for faster checking
 * - Only check relevant token subsets per state
 */

import { BenchmarkContext, type StateStats } from "./benchmark";

/**
 * Token classification for faster filtering
 */
enum TokenType {
  /** Only whitespace (space, newline, tab) */
  WHITESPACE_ONLY = 0,
  /** Only alphabetic characters (a-z, A-Z) */
  ALPHA_ONLY = 1,
  /** Starts with space, followed by alpha */
  SPACE_ALPHA = 2,
  /** Contains invalid characters - never valid */
  INVALID = 3,
}

/**
 * Tokenizer interface matching transformers.js
 */
type TokenizerLike = {
  decode: (
    tokens: Array<number | bigint>,
    options?: { skip_special_tokens?: boolean }
  ) => string;
};

/**
 * Tensor type matching transformers.js
 */
type Tensor = {
  data: Float32Array;
  dims: number[];
};

/**
 * Progress callback for long-running initialization
 */
export type InitProgressCallback = (progress: {
  phase: "caching" | "building" | "ready";
  percent: number;
  detail?: string;
}) => void;

/**
 * Trie node for prefix matching
 */
class TrieNode {
  public children: Map<string, TrieNode> = new Map();
  public isWordEnd: boolean = false;
}

/**
 * Trie for efficient prefix/word lookup
 */
class Trie {
  private root: TrieNode = new TrieNode();

  insert(word: string): void {
    let node = this.root;
    for (const char of word) {
      if (!node.children.has(char)) {
        node.children.set(char, new TrieNode());
      }
      node = node.children.get(char)!;
    }
    node.isWordEnd = true;
  }

  private traverse(prefix: string): TrieNode | null {
    let node: TrieNode | null = this.root;
    for (const char of prefix) {
      if (!node.children.has(char)) return null;
      node = node.children.get(char)!;
    }
    return node;
  }

  isPrefix(prefix: string): boolean {
    if (!prefix) return true; // Empty string is valid prefix (initial state)
    return this.traverse(prefix) !== null;
  }

  isWord(word: string): boolean {
    if (!word) return false;
    const node = this.traverse(word);
    return node !== null && node.isWordEnd;
  }

  /**
   * Check if a word can be extended (has children in the trie)
   */
  canExtend(prefix: string): boolean {
    const node = this.traverse(prefix);
    return node !== null && node.children.size > 0;
  }

  /**
   * Get all characters that can follow this prefix
   */
  getNextChars(prefix: string): string[] {
    const node = this.traverse(prefix);
    if (node === null) return [];
    return Array.from(node.children.keys());
  }

  /**
   * Get the root node's children (for initial state)
   */
  getRootNextChars(): string[] {
    return Array.from(this.root.children.keys());
  }

  /**
   * Get all words in the trie (for debugging)
   */
  getAllWords(): string[] {
    const words: string[] = [];
    const collect = (node: TrieNode, prefix: string): void => {
      if (node.isWordEnd) {
        words.push(prefix);
      }
      node.children.forEach((child, char) => {
        collect(child, prefix + char);
      });
    };
    collect(this.root, "");
    return words;
  }
}

/**
 * State in the constrained generation state machine.
 * State is identified by the current partial word prefix being built.
 */
interface State {
  /** Unique state ID */
  id: number;
  /** Current prefix (e.g., "", " h", " he", " hel", " hello") */
  prefix: string;
  /** Whether this prefix forms a complete word */
  isCompleteWord: boolean;
}

/**
 * Precomputed constrained logits processor.
 *
 * Guarantees:
 * - Only words from the wordset will be generated
 * - O(1) constraint checking per token during generation
 *
 * Tradeoff:
 * - Initialization takes 2-5 seconds to precompute all transitions
 */
export class PrecomputedConstrainedProcessor {
  private readonly trie: Trie;
  private readonly tokenizer: TokenizerLike;
  private readonly vocabSize: number;

  // Precomputed data structures
  private readonly tokenStrings: Map<number, string> = new Map();
  private readonly tokenTypes: Map<number, TokenType> = new Map();
  private readonly states: Map<string, State> = new Map();
  private readonly statesById: Map<number, State> = new Map();
  private readonly validTokens: Map<number, Set<number>> = new Map();
  private readonly nextState: Map<number, Map<number, number>> = new Map();

  // Token subsets by type for faster iteration
  private candidateTokenIds: number[] = []; // Tokens that could potentially be valid

  // First-character index: char -> list of token IDs starting with that char
  private tokensByFirstChar: Map<string, number[]> = new Map();
  // Special index for space-starting tokens
  private spaceStartingTokens: number[] = [];

  // Current generation state
  private currentStateId: number = 0;
  private generatedText: string = "";

  // Benchmarking (optional)
  private benchmark: BenchmarkContext | null = null;

  // State counter
  private stateCounter: number = 0;

  /**
   * Create a new precomputed processor.
   *
   * @param wordset - Array of allowed words
   * @param tokenizer - Tokenizer with decode() method
   * @param vocabSize - Size of vocabulary
   * @param options - Optional configuration
   */
  constructor(
    wordset: string[],
    tokenizer: TokenizerLike,
    vocabSize: number,
    options: {
      benchmark?: BenchmarkContext;
      onProgress?: InitProgressCallback;
    } = {}
  ) {
    this.tokenizer = tokenizer;
    this.vocabSize = vocabSize;
    this.benchmark = options.benchmark ?? null;
    const onProgress = options.onProgress;

    this.benchmark?.startInit();
    this.benchmark?.setVocabSize(vocabSize);
    this.benchmark?.setWordsetSize(wordset.length);

    // Build trie with space-prefixed words
    this.trie = new Trie();
    for (const word of wordset) {
      if (word.length > 0) {
        this.trie.insert(" " + word);
      }
    }

    // Cache all token strings
    this.benchmark?.startTokenCache();
    onProgress?.({ phase: "caching", percent: 0, detail: "Caching token strings..." });
    this.cacheTokenStrings(onProgress);
    this.benchmark?.endTokenCache();

    // Build state machine via BFS
    this.benchmark?.startStateMachineBuild();
    onProgress?.({ phase: "building", percent: 0, detail: "Building state machine..." });
    this.buildStateMachine(onProgress);
    this.benchmark?.endStateMachineBuild();

    // Collect state stats for benchmarking
    this.collectStateStats();

    this.benchmark?.endInit();
    onProgress?.({ phase: "ready", percent: 100, detail: "Ready" });
  }

  /**
   * Classify a token string into a type
   */
  private classifyToken(tokenStr: string): TokenType {
    if (tokenStr.length === 0) {
      return TokenType.INVALID;
    }

    let hasSpace = false;
    let hasAlpha = false;
    let hasInvalid = false;
    let startsWithSpace = false;

    for (let i = 0; i < tokenStr.length; i++) {
      const char = tokenStr[i]!;
      const isSpace = char === " " || char === "\n" || char === "\t";
      const isAlpha = /[a-zA-Z]/.test(char);

      if (i === 0 && isSpace) {
        startsWithSpace = true;
      }

      if (isSpace) {
        hasSpace = true;
      } else if (isAlpha) {
        hasAlpha = true;
      } else {
        // Numbers, punctuation, unicode, etc.
        hasInvalid = true;
      }
    }

    // Contains invalid characters - never valid
    if (hasInvalid) {
      return TokenType.INVALID;
    }

    // Only whitespace
    if (hasSpace && !hasAlpha) {
      return TokenType.WHITESPACE_ONLY;
    }

    // Starts with space, has alpha (e.g., " hello", " the")
    if (startsWithSpace && hasAlpha) {
      return TokenType.SPACE_ALPHA;
    }

    // Only alpha (e.g., "ello", "ing")
    if (hasAlpha && !hasSpace) {
      return TokenType.ALPHA_ONLY;
    }

    // Space in middle or end but not start - could still be valid
    // e.g., "hello " - but this is rare and complex, treat as potentially valid
    return TokenType.ALPHA_ONLY;
  }

  /**
   * Cache all token strings from the tokenizer and classify them
   */
  private cacheTokenStrings(onProgress?: InitProgressCallback): void {
    const reportInterval = Math.max(1, Math.floor(this.vocabSize / 20));
    let invalidCount = 0;

    for (let tokenId = 0; tokenId < this.vocabSize; tokenId++) {
      try {
        const decoded = this.tokenizer.decode([tokenId], {
          skip_special_tokens: false,
        });
        this.tokenStrings.set(tokenId, decoded);

        // Classify token
        const tokenType = this.classifyToken(decoded);
        this.tokenTypes.set(tokenId, tokenType);

        // Add to candidate list if potentially valid
        if (tokenType !== TokenType.INVALID) {
          this.candidateTokenIds.push(tokenId);
        } else {
          invalidCount++;
        }
      } catch {
        this.tokenStrings.set(tokenId, "");
        this.tokenTypes.set(tokenId, TokenType.INVALID);
        invalidCount++;
      }

      if (onProgress && tokenId % reportInterval === 0) {
        onProgress({
          phase: "caching",
          percent: Math.floor((tokenId / this.vocabSize) * 100),
          detail: `Cached ${tokenId}/${this.vocabSize} tokens`,
        });
      }
    }

    // Build first-character index for candidate tokens
    for (const tokenId of this.candidateTokenIds) {
      const tokenStr = this.tokenStrings.get(tokenId) ?? "";
      if (tokenStr.length === 0) continue;

      const firstChar = tokenStr[0]!.toLowerCase();

      // Index by first character
      if (firstChar === " " || firstChar === "\n" || firstChar === "\t") {
        this.spaceStartingTokens.push(tokenId);
      } else {
        if (!this.tokensByFirstChar.has(firstChar)) {
          this.tokensByFirstChar.set(firstChar, []);
        }
        this.tokensByFirstChar.get(firstChar)!.push(tokenId);
      }
    }

    console.log(
      `[PrecomputedProcessor] Token filtering: ${this.candidateTokenIds.length} candidates, ` +
      `${invalidCount} invalid (${((invalidCount / this.vocabSize) * 100).toFixed(1)}% filtered). ` +
      `First-char index: ${this.tokensByFirstChar.size} chars, ${this.spaceStartingTokens.length} space-starting`
    );
  }

  /**
   * Get or create a state for a given prefix
   */
  private getOrCreateState(prefix: string): State {
    let state = this.states.get(prefix);
    if (!state) {
      state = {
        id: this.stateCounter++,
        prefix,
        isCompleteWord: this.trie.isWord(prefix),
      };
      this.states.set(prefix, state);
      this.statesById.set(state.id, state);
      this.validTokens.set(state.id, new Set());
      this.nextState.set(state.id, new Map());
    }
    return state;
  }

  /**
   * Get candidate tokens to check for a given state.
   * Uses first-character indexing to dramatically reduce the search space.
   */
  private getCandidateTokensForState(prefix: string): number[] {
    const candidates: number[] = [];
    const seenTokens = new Set<number>();

    const addTokens = (tokens: number[]): void => {
      for (const tokenId of tokens) {
        if (!seenTokens.has(tokenId)) {
          seenTokens.add(tokenId);
          candidates.push(tokenId);
        }
      }
    };

    if (prefix === "") {
      // Initial state: only space-starting tokens are valid (words start with " ")
      addTokens(this.spaceStartingTokens);
    } else {
      // Get characters that can continue this prefix from the trie
      const nextChars = this.trie.getNextChars(prefix);
      for (const char of nextChars) {
        const charLower = char.toLowerCase();
        const tokens = this.tokensByFirstChar.get(charLower);
        if (tokens) {
          addTokens(tokens);
        }
      }

      // If prefix is a complete word, we can also start a new word
      // This means space-starting tokens are valid
      if (this.trie.isWord(prefix)) {
        addTokens(this.spaceStartingTokens);
      }
    }

    return candidates;
  }

  /**
   * Build the state machine by BFS exploration
   * Uses first-character indexing for dramatically faster exploration
   */
  private buildStateMachine(onProgress?: InitProgressCallback): void {
    // Create initial state (empty prefix, ready for first word)
    const initialState = this.getOrCreateState("");

    // BFS queue of states to process
    const queue: number[] = [initialState.id];
    const processed = new Set<number>();

    let statesProcessed = 0;
    let transitionsChecked = 0;
    let validTransitions = 0;

    while (queue.length > 0) {
      const stateId = queue.shift()!;
      if (processed.has(stateId)) continue;
      processed.add(stateId);

      const state = this.statesById.get(stateId)!;
      const stateValidTokens = this.validTokens.get(stateId)!;
      const stateNextState = this.nextState.get(stateId)!;

      // Get candidate tokens for this state (using first-char index)
      const candidateTokens = this.getCandidateTokensForState(state.prefix);

      for (const tokenId of candidateTokens) {
        transitionsChecked++;
        const tokenStr = this.tokenStrings.get(tokenId) ?? "";
        const result = this.computeTransition(state.prefix, tokenStr);

        if (result.isValid && result.nextPrefix !== null) {
          validTransitions++;
          stateValidTokens.add(tokenId);

          // Get or create the next state
          const nextStateObj = this.getOrCreateState(result.nextPrefix);
          stateNextState.set(tokenId, nextStateObj.id);

          // Add to queue if not yet processed
          if (!processed.has(nextStateObj.id)) {
            queue.push(nextStateObj.id);
          }
        }
      }

      statesProcessed++;
      if (onProgress && statesProcessed % 50 === 0) {
        onProgress({
          phase: "building",
          percent: Math.min(95, Math.floor((statesProcessed / Math.max(statesProcessed + queue.length, 1)) * 100)),
          detail: `Processed ${statesProcessed} states, ${queue.length} queued`,
        });
      }
    }

    console.log(
      `[PrecomputedProcessor] BFS complete: ${statesProcessed} states, ` +
      `${transitionsChecked} transitions checked, ${validTransitions} valid ` +
      `(${((validTransitions / transitionsChecked) * 100).toFixed(2)}%)`
    );
  }

  /**
   * Compute what happens when we append tokenStr to the current prefix.
   * Returns the resulting prefix (after any completed words) and whether it's valid.
   */
  private computeTransition(
    prefix: string,
    tokenStr: string
  ): { isValid: boolean; nextPrefix: string | null } {
    // Process the combined text
    let current = prefix + tokenStr;

    // Try to parse as valid sequence of complete words + optional prefix
    const result = this.parseText(current);
    return result;
  }

  /**
   * Parse text to extract the remaining prefix after any complete words.
   * Returns { isValid: true, nextPrefix } if valid, { isValid: false, nextPrefix: null } otherwise.
   *
   * Key insight: The state is the current prefix being built. When a prefix is both
   * a complete word AND can be continued (e.g., " the" can become " there"), we keep
   * the full prefix so continuations work.
   *
   * State transitions:
   * - "" + " the" -> " the" (prefix, also complete word, can extend to "there")
   * - " the" + "re" -> " there" (prefix)
   * - " the" + " w" -> " w" (word complete, start new word)
   * - " hello" -> "" (complete word, cannot extend, back to initial)
   */
  private parseText(text: string): { isValid: boolean; nextPrefix: string | null } {
    if (text === "") {
      return { isValid: true, nextPrefix: "" };
    }

    // Check if the whole text is a valid prefix
    if (this.trie.isPrefix(text)) {
      // If it's a complete word that CANNOT be extended, we could return ""
      // But we need to keep it as prefix if it CAN be extended (e.g., "the" -> "there")
      if (this.trie.isWord(text) && !this.trie.canExtend(text)) {
        // Complete word that can't be extended - back to initial state
        return { isValid: true, nextPrefix: "" };
      }
      // Either not a complete word (partial prefix), or can be extended
      return { isValid: true, nextPrefix: text };
    }

    // Text is not a valid prefix on its own. Try to parse as sequence of complete words.
    // We need to find the longest prefix that ends with a complete word, then recurse.
    const n = text.length;

    // Try from longest to shortest to find complete words
    for (let i = n; i >= 1; i--) {
      const candidate = text.slice(0, i);
      if (this.trie.isWord(candidate)) {
        const remainder = text.slice(i);

        // If no remainder, we've parsed everything
        if (remainder === "") {
          // Check if this word can be extended
          if (this.trie.canExtend(candidate)) {
            return { isValid: true, nextPrefix: candidate };
          }
          // Cannot extend - back to initial state
          return { isValid: true, nextPrefix: "" };
        }

        // Try parsing the remainder recursively
        const subResult = this.parseText(remainder);
        if (subResult.isValid) {
          return subResult;
        }
      }
    }

    // No valid parse found
    return { isValid: false, nextPrefix: null };
  }

  /**
   * Collect statistics about the state machine for benchmarking
   */
  private collectStateStats(): void {
    if (!this.benchmark) return;

    let totalValidTokens = 0;
    let minValid = Infinity;
    let maxValid = 0;

    this.validTokens.forEach((validSet) => {
      const count = validSet.size;
      totalValidTokens += count;
      minValid = Math.min(minValid, count);
      maxValid = Math.max(maxValid, count);
    });

    const stateCount = this.states.size;
    const stats: StateStats = {
      stateCount,
      avgValidTokensPerState: stateCount > 0 ? totalValidTokens / stateCount : 0,
      minValidTokensPerState: minValid === Infinity ? 0 : minValid,
      maxValidTokensPerState: maxValid,
      totalValidTransitions: totalValidTokens,
    };

    this.benchmark.setStateStats(stats);
  }

  /**
   * Reset the processor state for a new generation
   */
  reset(): void {
    this.currentStateId = 0;
    this.generatedText = "";
  }

  /**
   * Process logits by masking invalid tokens.
   * This is the hot path - must be O(1) per token.
   */
  process(inputIds: bigint[][], logits: Tensor): Tensor {
    if (inputIds.length === 0) return logits;

    // Get the sequence of tokens generated so far
    const batchIds = inputIds[0] ?? [];

    // Update state based on all tokens (in case we missed some)
    // This handles the case where transformers.js calls process() with accumulated tokens
    this.updateStateFromTokens(batchIds);

    // Get valid tokens for current state
    const validSet = this.validTokens.get(this.currentStateId);

    if (!validSet || validSet.size === 0) {
      // No valid tokens - this shouldn't happen with a properly constructed wordset
      console.error(
        `[PrecomputedProcessor] No valid tokens for state ${this.currentStateId}, ` +
          `prefix="${this.statesById.get(this.currentStateId)?.prefix ?? "?"}"`
      );
      return logits;
    }

    // Mask invalid tokens
    const vocab = logits.dims[logits.dims.length - 1] ?? this.vocabSize;
    for (let tokenId = 0; tokenId < vocab; tokenId++) {
      if (!validSet.has(tokenId)) {
        logits.data[tokenId] = -Infinity;
      }
    }

    return logits;
  }

  /**
   * Update internal state based on generated tokens.
   * Called internally by process().
   */
  private updateStateFromTokens(tokens: bigint[]): void {
    // Decode all tokens to get the full generated text
    if (tokens.length === 0) {
      this.currentStateId = 0;
      this.generatedText = "";
      return;
    }

    const ids = tokens.map((t) => Number(t));
    const decoded = this.tokenizer.decode(ids, { skip_special_tokens: true });

    // Only recompute if text changed
    if (decoded === this.generatedText) {
      return;
    }

    this.generatedText = decoded;

    // Parse the text to find current state
    const parseResult = this.parseText(decoded);
    if (parseResult.isValid && parseResult.nextPrefix !== null) {
      const state = this.states.get(parseResult.nextPrefix);
      if (state) {
        this.currentStateId = state.id;
      }
    }
  }

  /**
   * Get the processor function for transformers.js
   */
  getProcessor(): (inputIds: bigint[][], logits: Tensor) => Tensor {
    return (inputIds: bigint[][], logits: Tensor) => this.process(inputIds, logits);
  }

  /**
   * Get valid token IDs for the current state (for debugging/testing)
   */
  getValidTokenIds(): number[] {
    const validSet = this.validTokens.get(this.currentStateId);
    return validSet ? Array.from(validSet) : [];
  }

  /**
   * Get current state info (for debugging)
   */
  getCurrentStateInfo(): { stateId: number; prefix: string; isComplete: boolean } {
    const state = this.statesById.get(this.currentStateId);
    return {
      stateId: this.currentStateId,
      prefix: state?.prefix ?? "",
      isComplete: state?.isCompleteWord ?? false,
    };
  }

  /**
   * Get generated words from the accumulated text
   */
  getGeneratedWords(): string[] {
    // Split on spaces and filter empty strings
    return this.generatedText
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0);
  }

  /**
   * Get statistics about the state machine
   */
  getStats(): { stateCount: number; avgValidTokens: number } {
    let totalValid = 0;
    this.validTokens.forEach((validSet) => {
      totalValid += validSet.size;
    });
    return {
      stateCount: this.states.size,
      avgValidTokens: this.states.size > 0 ? totalValid / this.states.size : 0,
    };
  }
}

