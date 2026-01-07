type TokenizerLike = {
  decode: (
    tokens: Array<number | bigint>,
    options?: { skip_special_tokens?: boolean },
  ) => string;
};

/**
 * Simple trie for arbitrary strings (unicode-aware).
 */
class TrieNode {
  public children: Map<string, TrieNode>;
  public isWordEnd: boolean;

  constructor() {
    this.children = new Map();
    this.isWordEnd = false;
  }
}

class Trie {
  private root: TrieNode;

  constructor() {
    this.root = new TrieNode();
  }

  insert(word: string): void {
    let node = this.root;
    for (const char of word) {
      if (!node.children.has(char)) {
        node.children.set(char, new TrieNode());
      }
      node = node.children.get(char) as TrieNode;
    }
    node.isWordEnd = true;
  }

  private traverse(prefix: string): TrieNode | null {
    let node: TrieNode | null = this.root;
    for (const char of prefix) {
      if (!node.children.has(char)) return null;
      node = node.children.get(char) as TrieNode;
    }
    return node;
  }

  isPrefix(prefix: string): boolean {
    if (!prefix) return false;
    return this.traverse(prefix) !== null;
  }

  isWord(word: string): boolean {
    if (!word) return false;
    const node = this.traverse(word);
    return node !== null && node.isWordEnd;
  }
}

type Tensor = {
  data: Float32Array;
  dims: number[];
};

export type ConstrainedProcessorOptions = {
  debug?: boolean;
};

/**
 * Logits processor that mirrors the Python reference implementation.
 *
 * - Words are inserted into a trie with a leading space: " word".
 * - At each step, every token in the vocab is decoded and checked for validity
 *   against the trie and current prefix.
 * - No assumptions are made about character set; operates on arbitrary unicode.
 */
export class Gpt2ConstrainedLogitsProcessor {
  private readonly tokenizer: TokenizerLike;
  private readonly vocabSize: number;
  private readonly trie: Trie;
  private readonly tokenStrings: Map<number, string>;
  private readonly debug: boolean;

  constructor(
    wordset: string[],
    tokenizer: TokenizerLike,
    vocabSize: number,
    options: ConstrainedProcessorOptions = {},
  ) {
    this.tokenizer = tokenizer;
    this.vocabSize = vocabSize;
    this.debug = options.debug ?? false;

    this.trie = new Trie();
    // Insert all words with a leading space, matching Python reference
    for (const word of wordset) {
      this.trie.insert(` ${word}`);
    }

    this.tokenStrings = new Map();
    this.precomputeTokenStrings();
  }

  private precomputeTokenStrings(): void {
    for (let tokenId = 0; tokenId < this.vocabSize; tokenId++) {
      try {
        const decoded = this.tokenizer.decode([tokenId], {
          skip_special_tokens: false,
        });
        this.tokenStrings.set(tokenId, decoded);
      } catch {
        this.tokenStrings.set(tokenId, "");
      }
    }
  }

  public reset(): void {
    // no-op for now; kept for parity with previous API
  }

  private isValidContinuation(prefix: string, tokenStr: string): boolean {
    const candidate = prefix + tokenStr;

    if (this.trie.isPrefix(candidate)) return true;

    return this.canParseAsValidSequence(candidate);
  }

  private canParseAsValidSequence(text: string): boolean {
    if (!text) return true;
    if (this.trie.isPrefix(text)) return true;

    const n = text.length;
    for (let i = 1; i <= n; i++) {
      const potentialWord = text.slice(0, i);
      const remainder = text.slice(i);

      if (this.trie.isWord(potentialWord)) {
        if (!remainder) return true;
        if (this.trie.isPrefix(remainder)) return true;
        if (this.canParseAsValidSequence(remainder)) return true;
      }
    }

    return false;
  }

  private getCurrentPrefix(text: string): string {
    if (!text) return "";

    const n = text.length;
    let lastValidWordEnd = 0;
    let i = 0;

    while (i < n) {
      let bestMatchEnd = -1;
      for (let j = i + 1; j <= n; j++) {
        const candidate = text.slice(i, j);
        if (this.trie.isWord(candidate)) {
          const remainder = text.slice(j);
          if (!remainder || remainder.startsWith(" ") || this.trie.isPrefix(remainder)) {
            bestMatchEnd = j;
          }
        }
      }

      if (bestMatchEnd > 0) {
        lastValidWordEnd = bestMatchEnd;
        i = bestMatchEnd;
      } else {
        break;
      }
    }

    return text.slice(lastValidWordEnd);
  }

  private getValidTokenIds(currentPrefix: string): Set<number> {
    const valid = new Set<number>();
    for (let tokenId = 0; tokenId < this.vocabSize; tokenId++) {
      const tokenStr = this.tokenStrings.get(tokenId) ?? "";
      if (this.isValidContinuation(currentPrefix, tokenStr)) {
        valid.add(tokenId);
      }
    }
    return valid;
  }

  /**
   * transformers.js logits processor signature.
   */
  public process(input_ids: bigint[][], logits: Tensor): Tensor {
    if (input_ids.length === 0) return logits;

    const batchIds = input_ids[0] ?? [];
    const ids = batchIds.map((t) => Number(t));
    const decoded = this.tokenizer.decode(ids, { skip_special_tokens: true });
    const currentPrefix = this.getCurrentPrefix(decoded);
    const validIds = this.getValidTokenIds(currentPrefix);

    const vocab = logits.dims[logits.dims.length - 1] ?? this.vocabSize;
    for (let tokenId = 0; tokenId < vocab; tokenId++) {
      if (!validIds.has(tokenId)) {
        logits.data[tokenId] = -Infinity;
      }
    }

    if (this.debug && validIds.size === 0) {
      // eslint-disable-next-line no-console
      console.error(
        `[GPT2 Constrained] No valid tokens for prefix="${currentPrefix}" decoded="${decoded.slice(
          0,
          80,
        )}"`,
      );
    }

    return logits;
  }

  public getProcessor(): (input_ids: bigint[][], logits: Tensor) => Tensor {
    return (input_ids: bigint[][], logits: Tensor) => this.process(input_ids, logits);
  }
}

