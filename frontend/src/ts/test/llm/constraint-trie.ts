import { ConstraintStateId } from "./types";
import { isValidSurfaceForm } from "./surface-forms";

type ConstraintTrieNode = {
  children: Map<string, ConstraintStateId>;
  isWord: boolean;
  prefix: string;
};

export class ConstraintTrie {
  readonly rootStateId = 0;

  private readonly nodes: ConstraintTrieNode[] = [
    {
      children: new Map<string, ConstraintStateId>(),
      isWord: false,
      prefix: "",
    },
  ];
  private uniqueReachableWordByStateId: Array<string | null> = [];

  constructor(words: string[]) {
    if (words.length === 0) {
      throw new Error("ConstraintTrie requires at least one word");
    }

    for (const word of words) {
      if (!isValidSurfaceForm(word)) {
        throw new Error(
          `invalid surface form (must not contain spaces): ${word}`,
        );
      }

      this.addWord(word);
    }

    this.precomputeUniqueReachableWords();
  }

  getStateCount(): number {
    return this.nodes.length;
  }

  getPrefix(stateId: ConstraintStateId): string {
    return this.getNode(stateId).prefix;
  }

  isWordState(stateId: ConstraintStateId): boolean {
    return this.getNode(stateId).isWord;
  }

  // If every forward path of word-chars from this state reaches the same
  // terminal word, returns that word. Otherwise null. Precomputed at trie
  // construction time with a single bottom-up DFS — O(trie states) total.
  // Uses only the char-level trie structure, does not materialize the
  // (much more expensive) token-level transitions.
  getUniqueReachableWord(stateId: ConstraintStateId): string | null {
    return this.uniqueReachableWordByStateId[stateId] ?? null;
  }

  private precomputeUniqueReachableWords(): void {
    // iterative post-order DFS; computes each node's uniqueReachable
    // after all its children have been computed.
    this.uniqueReachableWordByStateId = new Array<string | null>(
      this.nodes.length,
    ).fill(null);

    type Frame = { stateId: ConstraintStateId; childIndex: number };
    const stack: Frame[] = [{ stateId: this.rootStateId, childIndex: 0 }];
    // pre-build child order arrays to get deterministic iteration + O(1) indexing
    const childOrder: ConstraintStateId[][] = this.nodes.map((node) =>
      Array.from(node.children.values()),
    );

    while (stack.length > 0) {
      const frame = stack[stack.length - 1] as Frame;
      const node = this.getNode(frame.stateId);
      const children = childOrder[frame.stateId] ?? [];

      if (frame.childIndex < children.length) {
        const childId = children[frame.childIndex] as ConstraintStateId;
        frame.childIndex++;
        stack.push({ stateId: childId, childIndex: 0 });
        continue;
      }

      // All children processed; compute this node's value.
      if (frame.stateId === this.rootStateId) {
        // root is the natural branching point — any word reachable.
        this.uniqueReachableWordByStateId[frame.stateId] = null;
      } else {
        let unique: string | null = node.isWord ? node.prefix : null;
        let ambiguous = false;
        for (const childId of children) {
          const childWord = this.uniqueReachableWordByStateId[childId] ?? null;
          if (childWord === null) {
            ambiguous = true;
            break;
          }
          if (unique === null) {
            unique = childWord;
          } else if (unique !== childWord) {
            ambiguous = true;
            break;
          }
        }
        this.uniqueReachableWordByStateId[frame.stateId] = ambiguous
          ? null
          : unique;
      }

      stack.pop();
    }
  }

  consumeChar(
    stateId: ConstraintStateId,
    char: string,
  ): ConstraintStateId | null {
    const node = this.getNode(stateId);
    const directTransition = node.children.get(char);

    if (directTransition !== undefined) {
      return directTransition;
    }

    if (char === " " && node.isWord) {
      return this.rootStateId;
    }

    return null;
  }

  consumeText(
    stateId: ConstraintStateId,
    text: string,
  ): ConstraintStateId | null {
    let currentStateId = stateId;

    for (const char of text) {
      const nextStateId = this.consumeChar(currentStateId, char);

      if (nextStateId === null) {
        return null;
      }

      currentStateId = nextStateId;
    }

    return currentStateId;
  }

  private addWord(word: string): void {
    if (word.length === 0) {
      throw new Error("ConstraintTrie words must be non-empty");
    }

    let currentStateId = this.rootStateId;

    for (const char of word) {
      const currentNode = this.getNode(currentStateId);
      let nextStateId = currentNode.children.get(char);

      if (nextStateId === undefined) {
        nextStateId = this.nodes.length;
        this.nodes.push({
          children: new Map<string, ConstraintStateId>(),
          isWord: false,
          prefix: `${currentNode.prefix}${char}`,
        });
        currentNode.children.set(char, nextStateId);
      }

      currentStateId = nextStateId;
    }

    this.getNode(currentStateId).isWord = true;
  }

  private getNode(stateId: ConstraintStateId): ConstraintTrieNode {
    const node = this.nodes[stateId];

    if (node === undefined) {
      throw new Error(`Unknown constraint trie state: ${stateId}`);
    }

    return node;
  }
}
