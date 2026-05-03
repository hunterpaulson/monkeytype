import { ConstraintTrie } from "./constraint-trie";
import {
  ConstraintEngineStats,
  ConstraintStateId,
  ConstraintStateProfile,
  DecodedToken,
  TokenId,
  TokenizerAdapter,
} from "./types";

type PrecomputedState = {
  validTokenIds: TokenId[];
  transitions: Map<TokenId, ConstraintStateId>;
};

type TokenTrieNode = {
  children: Map<string, number>;
  tokenIds: TokenId[];
};

type TokenIndex = {
  rootNodeId: number;
  nodes: TokenTrieNode[];
};

export function decodeTokenizerVocabulary(
  tokenizer: TokenizerAdapter,
): DecodedToken[] {
  const decodedTokens: DecodedToken[] = [];

  for (let tokenId = 0; tokenId < tokenizer.getVocabSize(); tokenId++) {
    decodedTokens.push({
      id: tokenId,
      text: tokenizer.decodeToken(tokenId),
    });
  }

  return decodedTokens;
}

export class ConstraintEngine {
  readonly rootStateId: ConstraintStateId;

  private readonly trie: ConstraintTrie;
  private readonly decodedTokens: DecodedToken[];
  private readonly tokenIndex: TokenIndex;
  private readonly states: Array<PrecomputedState | undefined>;
  private readonly stateProfiles: ConstraintStateProfile[];

  constructor(words: string[], decodedTokens: DecodedToken[]) {
    this.trie = new ConstraintTrie(words);
    this.rootStateId = this.trie.rootStateId;
    this.decodedTokens = decodedTokens;
    this.tokenIndex = buildTokenIndex(decodedTokens, words);
    this.states = new Array<PrecomputedState | undefined>(
      this.trie.getStateCount(),
    );
    this.stateProfiles = Array.from(
      { length: this.trie.getStateCount() },
      (_, stateId) => {
        return {
          stateId,
          prefix: this.trie.getPrefix(stateId),
          materialized: false,
          materializeMs: 0,
          validTokenCount: 0,
          accessCount: 0,
        };
      },
    );
  }

  getStateCount(): number {
    return this.states.length;
  }

  // If every forward path of word-chars from this state reaches the same
  // terminal word, returns that word. Otherwise null. Precomputed on the
  // underlying trie in O(trie states). Used by the word-level frequency
  // penalty to penalize mid-word prefixes whose completion is already
  // determined (e.g. "pi" when "pioneer" is the only pi-word in the wordset).
  getUniqueReachableWord(stateId: ConstraintStateId): string | null {
    return this.trie.getUniqueReachableWord(stateId);
  }

  getStatePrefix(stateId: ConstraintStateId): string {
    return this.trie.getPrefix(stateId);
  }

  canTerminate(stateId: ConstraintStateId): boolean {
    return this.trie.isWordState(stateId);
  }

  getValidTokenIds(stateId: ConstraintStateId): readonly TokenId[] {
    return this.getState(stateId).validTokenIds;
  }

  consumeText(
    stateId: ConstraintStateId,
    text: string,
  ): ConstraintStateId | null {
    return this.trie.consumeText(stateId, text);
  }

  getNextState(
    stateId: ConstraintStateId,
    tokenId: TokenId,
  ): ConstraintStateId | null {
    return this.getState(stateId).transitions.get(tokenId) ?? null;
  }

  // Returns true if sampling this token would either (a) directly complete one
  // of the banned words, or (b) land us in a non-terminal state from which every
  // reachable completion is a banned word ("dead-end trap"). Used to mirror
  // monkeytype's "don't repeat the last N words" rule inside constrained
  // decoding, and to prevent walking into prefixes whose only completion is a
  // recently emitted word (e.g. "pi" when "pioneer" is the only pi-word and
  // pioneer is in the banned set).
  tokenLeadsToBannedWord(
    stateId: ConstraintStateId,
    tokenId: TokenId,
    bannedWords: ReadonlySet<string>,
    memo: Map<ConstraintStateId, boolean>,
  ): boolean {
    if (bannedWords.size === 0) {
      return false;
    }

    const nextStateId = this.getNextState(stateId, tokenId);

    if (nextStateId === null) {
      return false;
    }

    if (this.canTerminate(nextStateId)) {
      return bannedWords.has(this.getStatePrefix(nextStateId));
    }

    return this.stateOnlyReachesBannedWords(nextStateId, bannedWords, memo);
  }

  // Legacy single-step check. Kept for tests that exercise the direct completion
  // case explicitly. Prefer tokenLeadsToBannedWord in the generation loop.
  tokenCompletesBannedWord(
    stateId: ConstraintStateId,
    tokenId: TokenId,
    bannedWords: ReadonlySet<string>,
  ): boolean {
    if (bannedWords.size === 0) {
      return false;
    }

    const nextStateId = this.getNextState(stateId, tokenId);

    if (nextStateId === null) {
      return false;
    }

    if (!this.canTerminate(nextStateId)) {
      return false;
    }

    return bannedWords.has(this.getStatePrefix(nextStateId));
  }

  // Returns true iff, starting from stateId and following any sequence of valid
  // token transitions, every reachable word-terminal has a prefix in bannedWords.
  //
  // Terminal states and the root state are never dead-ends: a terminal can
  // space-transition back to root, and root can reach any word in the wordset,
  // so (assuming the wordset has at least one unbanned word, which our caller
  // guarantees) an unbanned terminal is always reachable from them.
  //
  // Dead-ends only exist at non-terminal, non-root states whose every forward
  // token transition commits to a banned word. Results are memoized per call
  // because the banned set changes across generation steps.
  stateOnlyReachesBannedWords(
    stateId: ConstraintStateId,
    bannedWords: ReadonlySet<string>,
    memo: Map<ConstraintStateId, boolean>,
  ): boolean {
    if (stateId === this.rootStateId) {
      return false;
    }

    if (this.canTerminate(stateId)) {
      return false;
    }

    const cached = memo.get(stateId);

    if (cached !== undefined) {
      return cached;
    }

    // tentatively mark dead-end to break any accidental cycle; will overwrite
    // if we find an escape route.
    memo.set(stateId, true);

    const state = this.getState(stateId);

    for (const tokenId of state.validTokenIds) {
      const nextStateId = state.transitions.get(tokenId);

      if (nextStateId === undefined) {
        continue;
      }

      if (this.canTerminate(nextStateId)) {
        const prefix = this.getStatePrefix(nextStateId);
        if (!bannedWords.has(prefix)) {
          memo.set(stateId, false);
          return false;
        }
        // banned terminal reached, but a terminal can space-out to root, so
        // from the CALLER's perspective the banned word gets emitted. we only
        // treat this as "escape" if the caller wanted to pass through. for the
        // dead-end check we care about the prefix we're emitting NOW, which is
        // banned — so keep looking for an unbanned-completion escape.
        continue;
      }

      if (nextStateId === this.rootStateId) {
        // reached root via a transition that did NOT pass through a banned
        // terminal-only path — safe.
        memo.set(stateId, false);
        return false;
      }

      if (!this.stateOnlyReachesBannedWords(nextStateId, bannedWords, memo)) {
        memo.set(stateId, false);
        return false;
      }
    }

    memo.set(stateId, true);
    return true;
  }

  getStats(): ConstraintEngineStats {
    this.materializeAllStates();

    const validTokenCount = this.states.reduce((sum, state) => {
      return sum + (state?.validTokenIds.length ?? 0);
    }, 0);

    return {
      stateCount: this.states.length,
      tokenCount: this.decodedTokens.length,
      averageValidTokensPerState:
        this.states.length === 0 ? 0 : validTokenCount / this.states.length,
    };
  }

  getStateProfile(stateId: ConstraintStateId): ConstraintStateProfile {
    const stateProfile = this.stateProfiles[stateId];

    if (stateProfile === undefined) {
      throw new Error(`Unknown constraint engine state profile: ${stateId}`);
    }

    return { ...stateProfile };
  }

  getStateProfiles(): ConstraintStateProfile[] {
    return this.stateProfiles.map((stateProfile) => ({ ...stateProfile }));
  }

  materializeAllStates(): void {
    for (let stateId = 0; stateId < this.trie.getStateCount(); stateId++) {
      this.getState(stateId);
    }
  }

  private collectTransitionsForState(
    stateId: ConstraintStateId,
  ): Map<TokenId, ConstraintStateId> {
    const transitions = new Map<TokenId, ConstraintStateId>();
    const stack: Array<{
      tokenNodeId: number;
      trieStateId: ConstraintStateId;
    }> = [
      {
        tokenNodeId: this.tokenIndex.rootNodeId,
        trieStateId: stateId,
      },
    ];

    while (stack.length > 0) {
      const current = stack.pop();

      if (current === undefined) {
        continue;
      }

      const tokenNode = this.tokenIndex.nodes[current.tokenNodeId];

      if (tokenNode === undefined) {
        continue;
      }

      for (const tokenId of tokenNode.tokenIds) {
        transitions.set(tokenId, current.trieStateId);
      }

      for (const [char, childTokenNodeId] of tokenNode.children) {
        const nextTrieStateId = this.trie.consumeChar(
          current.trieStateId,
          char,
        );

        if (nextTrieStateId === null) {
          continue;
        }

        stack.push({
          tokenNodeId: childTokenNodeId,
          trieStateId: nextTrieStateId,
        });
      }
    }

    return transitions;
  }

  private getState(stateId: ConstraintStateId): PrecomputedState {
    const stateProfile = this.stateProfiles[stateId];

    if (stateProfile === undefined) {
      throw new Error(`Unknown constraint engine state profile: ${stateId}`);
    }

    stateProfile.accessCount++;

    let state = this.states[stateId];

    if (state === undefined) {
      const materializeStart = performance.now();
      const transitions = this.collectTransitionsForState(stateId);
      state = {
        validTokenIds: Array.from(transitions.keys()).sort((left, right) => {
          return left - right;
        }),
        transitions,
      };
      this.states[stateId] = state;
      stateProfile.materialized = true;
      stateProfile.materializeMs = performance.now() - materializeStart;
      stateProfile.validTokenCount = state.validTokenIds.length;
    }

    return state;
  }
}

function buildTokenIndex(
  decodedTokens: DecodedToken[],
  words: string[],
): TokenIndex {
  const allowedChars = buildAllowedCharSet(words);
  const nodes: TokenTrieNode[] = [
    {
      children: new Map<string, number>(),
      tokenIds: [],
    },
  ];

  for (const token of decodedTokens) {
    if (!isGloballyValidTokenText(token.text, allowedChars)) {
      continue;
    }

    let currentNodeId = 0;

    for (const char of token.text) {
      const currentNode = nodes[currentNodeId];

      if (currentNode === undefined) {
        throw new Error(`Unknown token trie node: ${currentNodeId}`);
      }

      let nextNodeId = currentNode.children.get(char);

      if (nextNodeId === undefined) {
        nextNodeId = nodes.length;
        nodes.push({
          children: new Map<string, number>(),
          tokenIds: [],
        });
        currentNode.children.set(char, nextNodeId);
      }

      currentNodeId = nextNodeId;
    }

    const terminalNode = nodes[currentNodeId];

    if (terminalNode === undefined) {
      throw new Error(`Unknown token trie node: ${currentNodeId}`);
    }

    terminalNode.tokenIds.push(token.id);
  }

  return {
    rootNodeId: 0,
    nodes,
  };
}

function buildAllowedCharSet(words: string[]): Set<string> {
  const allowedChars = new Set<string>();

  for (const word of words) {
    for (const char of word) {
      allowedChars.add(char);
    }
  }

  return allowedChars;
}

function isGloballyValidTokenText(
  text: string,
  allowedChars: Set<string>,
): boolean {
  if (text.length === 0) {
    return false;
  }

  let previousCharWasSpace = false;

  for (const char of text) {
    if (char === " ") {
      if (previousCharWasSpace) {
        return false;
      }

      previousCharWasSpace = true;
      continue;
    }

    if (!allowedChars.has(char)) {
      return false;
    }

    previousCharWasSpace = false;
  }

  return true;
}
