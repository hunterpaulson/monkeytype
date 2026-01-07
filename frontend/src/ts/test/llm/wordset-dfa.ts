/**
 * Deterministic Finite Automaton (DFA) for Wordset Constraint
 *
 * This implements Character Prefix Conditioning (CPC) as described in
 * https://cursor.com/blog/cpc
 *
 * Key differences from the trie approach:
 * 1. Handles actual characters (not just lowercase a-z) - supports punctuation, capitalization, etc.
 * 2. Precomputes token-to-state transitions for O(1) lookup during generation
 * 3. Tracks state explicitly (current partial word, word boundaries, etc.)
 *
 * The DFA states represent:
 * - Current partial word being built
 * - Whether we're at a word boundary (ready for new word)
 * - Whether the current partial word is a complete word in the wordset
 */

type DFAState = {
  /** Current partial word (normalized for matching) */
  partialWord: string;
  /** Whether this state represents a complete word */
  isCompleteWord: boolean;
  /** Whether we're at a word boundary (after space, ready for new word) */
  atWordBoundary: boolean;
  /** Unique state ID for efficient lookup */
  id: number;
};

/**
 * Transition result from processing a token string
 */
type TransitionResult = {
  /** The resulting state after processing the token */
  nextState: DFAState | null;
  /** Whether this transition is valid */
  isValid: boolean;
  /** If a word was completed during this transition */
  completedWord: string | null;
};

/**
 * Wordset DFA for constrained generation
 *
 * Usage:
 * ```typescript
 * const dfa = new WordsetDFA(["hello", "world", "test"]);
 * const transitions = dfa.precomputeTokenTransitions(tokenizer, vocabSize);
 * const nextState = transitions[stateId][tokenId];
 * ```
 */
export class WordsetDFA {
  private words: Set<string>;
  private wordPrefixes: Set<string>;
  private states: Map<string, DFAState> = new Map();
  private statesById: Map<number, DFAState> = new Map();
  private stateCounter: number = 0;
  private readonly minWordLength: number;
  private readonly maxWordLength: number;

  /**
   * Normalize a word for matching (lowercase, but preserve structure)
   * We keep the original characters but normalize case for matching
   */
  private normalizeWord(word: string): string {
    return word.toLowerCase();
  }

  /**
   * Extract alphabetic characters from a string (for word matching)
   * This preserves the word structure while ignoring punctuation
   */
  private extractWordChars(str: string): string {
    return str
      .toLowerCase()
      .split("")
      .filter((c) => /[a-z]/.test(c))
      .join("");
  }

  /**
   * Check if a character is a word boundary (space, newline, tab)
   */
  private isWordBoundary(char: string): boolean {
    return char === " " || char === "\n" || char === "\t";
  }

  /**
   * Check if a character is alphabetic
   */
  private isAlphabetic(char: string): boolean {
    return /[a-zA-Z]/.test(char);
  }

  constructor(
    words: string[],
    config: { minWordLength?: number; maxWordLength?: number } = {}
  ) {
    this.minWordLength = config.minWordLength ?? 1;
    this.maxWordLength = config.maxWordLength ?? 15;

    // Build word set and prefix set
    this.words = new Set();
    this.wordPrefixes = new Set();

    for (const word of words) {
      const normalized = this.normalizeWord(word);
      const wordChars = this.extractWordChars(normalized);

      if (wordChars.length === 0) continue;
      if (wordChars.length < this.minWordLength) continue;
      if (wordChars.length > this.maxWordLength) continue;

      this.words.add(wordChars);

      // Add all prefixes
      for (let i = 1; i <= wordChars.length; i++) {
        this.wordPrefixes.add(wordChars.substring(0, i));
      }
    }

    // Create initial state (empty, at word boundary, not a complete word)
    this.getOrCreateState("", false, true);
  }

  /**
   * Get or create a DFA state
   */
  private getOrCreateState(
    partialWord: string,
    isCompleteWord: boolean,
    atWordBoundary: boolean
  ): DFAState {
    const key = `${partialWord}|${isCompleteWord}|${atWordBoundary}`;
    let state = this.states.get(key);

    if (!state) {
      state = {
        partialWord,
        isCompleteWord,
        atWordBoundary,
        id: this.stateCounter++,
      };
      this.states.set(key, state);
      this.statesById.set(state.id, state);
    }

    return state;
  }

  /**
   * Get state by ID
   */
  public getStateById(stateId: number): DFAState | null {
    return this.statesById.get(stateId) ?? null;
  }

  /**
   * Get the initial state (empty, at word boundary)
   */
  public getInitialState(): DFAState {
    const state = this.states.get("|false|true");
    if (!state) {
      throw new Error("Initial state not found");
    }
    return state;
  }

  /**
   * Process a token string and transition from current state to next state
   */
  public transition(
    currentState: DFAState,
    tokenString: string
  ): TransitionResult {
    let partialWord = currentState.partialWord;
    let atWordBoundary = currentState.atWordBoundary;
    let completedWord: string | null = null;
    
    // Track if we encountered any meaningful characters (word boundary or alphabetic)
    let hasWordBoundary = false;
    let hasAlphabetic = false;

    // Track if token contains any non-boundary, non-alphabetic characters
    let hasOtherChars = false;

    // Process each character in the token
    for (const char of tokenString) {
      if (this.isWordBoundary(char)) {
        hasWordBoundary = true;
        // Word boundary encountered
        if (partialWord.length > 0) {
          // Check if we completed a valid word
          if (this.words.has(partialWord)) {
            completedWord = partialWord;
          } else {
            // Incomplete word - invalid transition
            return {
              nextState: null,
              isValid: false,
              completedWord: null,
            };
          }
        }
        // Reset for new word
        partialWord = "";
        atWordBoundary = true;
      } else if (this.isAlphabetic(char)) {
        hasAlphabetic = true;
        // Alphabetic character - continue building word
        const newPartial = partialWord + char.toLowerCase();

        // Check if this is a valid prefix
        if (!this.wordPrefixes.has(newPartial)) {
          // Invalid prefix - invalid transition
          return {
            nextState: null,
            isValid: false,
            completedWord: null,
          };
        }

        // Check max length
        if (newPartial.length > this.maxWordLength) {
          return {
            nextState: null,
            isValid: false,
            completedWord: null,
          };
        }

        partialWord = newPartial;
        atWordBoundary = false;
      } else {
        // Non-alphabetic, non-boundary character (numbers, punctuation, etc.)
        hasOtherChars = true;
      }
    }

    // If we're at a word boundary after processing:
    // - Token must start a valid word prefix (hasAlphabetic) OR
    // - Token must be pure word boundaries (hasWordBoundary && !hasOtherChars && !hasAlphabetic)
    // - Tokens with word boundaries + other non-alphabetic chars are invalid
    if (atWordBoundary && partialWord.length === 0) {
      if (!hasAlphabetic) {
        // No alphabetic characters - only valid if token is pure word boundaries
        if (!hasWordBoundary || hasOtherChars) {
          return {
            nextState: null,
            isValid: false,
            completedWord: null,
          };
        }
      }
    }

    // Determine if the resulting partial word is complete
    const isCompleteWord =
      partialWord.length > 0 && this.words.has(partialWord);

    // Create or get the next state
    const nextState = this.getOrCreateState(
      partialWord,
      isCompleteWord,
      atWordBoundary
    );

    return {
      nextState,
      isValid: true,
      completedWord,
    };
  }

  /**
   * Precompute transitions for all token IDs
   * This is the key optimization: O(1) lookup during generation
   *
   * We need to explore the state space by discovering new states as we transition.
   * This uses a breadth-first exploration: start with initial state, try all tokens,
   * and when we discover new states, add them to the work queue.
   *
   * @param tokenizer - Tokenizer with decode function
   * @param vocabSize - Vocabulary size
   * @returns Map from state ID to Map from token ID to next state ID (or null if invalid)
   */
  public precomputeTokenTransitions(
    tokenizer: {
      decode: (
        tokens: bigint[] | number[],
        options?: { skip_special_tokens?: boolean }
      ) => string;
    },
    vocabSize: number
  ): Map<number, Map<number, number | null>> {
    // Build token string cache
    const tokenStrings = new Map<number, string>();
    for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
      try {
        const decoded = tokenizer.decode([tokenId], {
          skip_special_tokens: false,
        });
        tokenStrings.set(tokenId, decoded);
      } catch {
        tokenStrings.set(tokenId, "");
      }
    }

    // Precompute transitions: stateId -> tokenId -> nextStateId | null
    const transitions = new Map<number, Map<number, number | null>>();
    const statesToProcess = new Set<number>();
    const processedStates = new Set<number>();

    // Start with initial state
    const initialState = this.getInitialState();
    statesToProcess.add(initialState.id);
    transitions.set(initialState.id, new Map());

    // Breadth-first exploration of state space
    let validTransitions = 0;
    let totalTransitions = 0;

    while (statesToProcess.size > 0) {
      // Get next state to process
      const stateId = statesToProcess.values().next().value as number | undefined;
      if (stateId === undefined) break;
      
      statesToProcess.delete(stateId);
      processedStates.add(stateId);

      const state = this.statesById.get(stateId);
      if (!state) continue;

      const stateTransitions = transitions.get(stateId);
      if (!stateTransitions) {
        transitions.set(stateId, new Map());
        continue;
      }

      // Try all tokens from this state
      for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
        totalTransitions++;
        const tokenString = tokenStrings.get(tokenId) ?? "";
        const result = this.transition(state, tokenString);

        if (result.isValid && result.nextState) {
          stateTransitions.set(tokenId, result.nextState.id);
          validTransitions++;

          // If we discovered a new state, add it to the work queue
          if (!processedStates.has(result.nextState.id)) {
            statesToProcess.add(result.nextState.id);
            if (!transitions.has(result.nextState.id)) {
              transitions.set(result.nextState.id, new Map());
            }
          }
        } else {
          stateTransitions.set(tokenId, null);
        }
      }
    }

    console.log(
      `[DFA] Precomputed transitions for ${this.states.size} states and ${vocabSize} tokens: ` +
        `${validTransitions}/${totalTransitions} valid transitions (${(
          (validTransitions / totalTransitions) *
          100
        ).toFixed(2)}%)`
    );

    return transitions;
  }

  /**
   * Get number of words in the wordset
   */
  public size(): number {
    return this.words.size;
  }

  /**
   * Get number of states in the DFA
   */
  public getStateCount(): number {
    return this.states.size;
  }
}

