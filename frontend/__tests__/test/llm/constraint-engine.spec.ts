import { describe, expect, it } from "vitest";
import {
  ConstraintEngine,
  decodeTokenizerVocabulary,
} from "../../../src/ts/test/llm/constraint-engine";

describe("constraint engine", () => {
  it("precomputes valid token transitions for root and mid-word states", () => {
    const engine = new ConstraintEngine(
      ["the", "there", "world"],
      [
        { id: 0, text: "the" },
        { id: 1, text: "there" },
        { id: 2, text: "re" },
        { id: 3, text: " world" },
        { id: 4, text: "w" },
        { id: 5, text: "orld" },
        { id: 6, text: "the world" },
        { id: 7, text: "x" },
        { id: 8, text: "" },
      ],
    );

    expect(engine.getValidTokenIds(engine.rootStateId)).toEqual([0, 1, 4, 6]);

    const theState = engine.getNextState(engine.rootStateId, 0);
    expect(theState).not.toBeNull();
    expect(engine.getStatePrefix(theState as number)).toBe("the");
    expect(engine.canTerminate(theState as number)).toBe(true);
    expect(engine.getValidTokenIds(theState as number)).toEqual([2, 3]);

    const worldFromSpan = engine.getNextState(engine.rootStateId, 6);
    expect(worldFromSpan).not.toBeNull();
    expect(engine.getStatePrefix(worldFromSpan as number)).toBe("world");
    expect(engine.canTerminate(worldFromSpan as number)).toBe(true);

    expect(engine.getNextState(engine.rootStateId, 3)).toBeNull();
    expect(engine.getNextState(engine.rootStateId, 8)).toBeNull();
    expect(engine.canTerminate(engine.rootStateId)).toBe(false);
  });

  it("rejects invalid surface forms in the constructor", () => {
    expect(() => {
      new ConstraintEngine(["hello", "two words"], [{ id: 0, text: "hello" }]);
    }).toThrow("invalid surface form (must not contain spaces)");
  });

  it("decodes tokenizer vocab through the adapter interface", () => {
    const decoded = decodeTokenizerVocabulary({
      getVocabSize() {
        return 3;
      },
      decodeToken(tokenId) {
        return ["a", "b", "c"][tokenId] as string;
      },
    });

    expect(decoded).toEqual([
      { id: 0, text: "a" },
      { id: 1, text: "b" },
      { id: 2, text: "c" },
    ]);
  });

  describe("tokenCompletesBannedWord", () => {
    const engine = new ConstraintEngine(
      ["the", "there", "world"],
      [
        { id: 0, text: "the" },
        { id: 1, text: "there" },
        { id: 2, text: "re" },
        { id: 3, text: " world" },
        { id: 4, text: "w" },
        { id: 5, text: "orld" },
        { id: 6, text: "the world" },
        { id: 7, text: "x" },
      ],
    );

    it("returns false when banned set is empty", () => {
      expect(
        engine.tokenCompletesBannedWord(engine.rootStateId, 0, new Set()),
      ).toBe(false);
    });

    it("returns true when token lands on a banned word-terminal state", () => {
      expect(
        engine.tokenCompletesBannedWord(
          engine.rootStateId,
          0,
          new Set(["the"]),
        ),
      ).toBe(true);
    });

    it("returns false when token lands on a non-banned terminal state", () => {
      expect(
        engine.tokenCompletesBannedWord(
          engine.rootStateId,
          0,
          new Set(["world"]),
        ),
      ).toBe(false);
    });

    it("returns false when token is invalid from this state", () => {
      expect(
        engine.tokenCompletesBannedWord(
          engine.rootStateId,
          7,
          new Set(["the"]),
        ),
      ).toBe(false);
    });

    it("returns false when token lands mid-word (not terminal)", () => {
      expect(
        engine.tokenCompletesBannedWord(
          engine.rootStateId,
          4,
          new Set(["world"]),
        ),
      ).toBe(false);
    });

    it("returns true for a multi-word token that lands on a banned terminal", () => {
      expect(
        engine.tokenCompletesBannedWord(
          engine.rootStateId,
          6,
          new Set(["world"]),
        ),
      ).toBe(true);
    });

    it("catches same-word emission from a word-terminal state", () => {
      // simulates the ' pioneer pioneer' back-to-back case: after emitting
      // ' pioneer' we are at the 'world' terminal state (using 'world' as a
      // stand-in), and sampling ' world' again should be flagged when 'world'
      // is in the banned set.
      const wordEngine = new ConstraintEngine(
        ["world"],
        [
          { id: 0, text: " world" },
          { id: 1, text: "world" },
        ],
      );

      const worldTerminal = wordEngine.getNextState(wordEngine.rootStateId, 1);
      expect(worldTerminal).not.toBeNull();
      expect(wordEngine.canTerminate(worldTerminal as number)).toBe(true);

      expect(
        wordEngine.tokenCompletesBannedWord(
          worldTerminal as number,
          0,
          new Set(["world"]),
        ),
      ).toBe(true);
    });
  });

  describe("tokenLeadsToBannedWord (prefix-aware)", () => {
    // wordset where "pi" is a dead-end prefix: "pioneer" is the only pi-word.
    // "pr" has multiple options (problem, press) so walking into "pr" is safe.
    const engine = new ConstraintEngine(
      ["pioneer", "problem", "press"],
      [
        { id: 0, text: "p" },
        { id: 1, text: "i" },
        { id: 2, text: "oneer" },
        { id: 3, text: "pioneer" },
        { id: 4, text: "r" },
        { id: 5, text: "oblem" },
        { id: 6, text: "ess" },
        { id: 7, text: "pi" },
        { id: 8, text: "pr" },
      ],
    );

    it("flags the direct completion of a banned word", () => {
      const memo = new Map<number, boolean>();
      expect(
        engine.tokenLeadsToBannedWord(
          engine.rootStateId,
          3,
          new Set(["pioneer"]),
          memo,
        ),
      ).toBe(true);
    });

    it("flags a dead-end prefix transition when pioneer is banned", () => {
      // from root, token "pi" lands at "pi" state, which only leads to pioneer.
      const memo = new Map<number, boolean>();
      expect(
        engine.tokenLeadsToBannedWord(
          engine.rootStateId,
          7,
          new Set(["pioneer"]),
          memo,
        ),
      ).toBe(true);
    });

    it("does not flag a branching prefix when one banned word has alternatives", () => {
      // from root, token "pr" lands at "pr" state, which branches to "problem"
      // or "press". even with "problem" banned, "press" is reachable.
      const memo = new Map<number, boolean>();
      expect(
        engine.tokenLeadsToBannedWord(
          engine.rootStateId,
          8,
          new Set(["problem"]),
          memo,
        ),
      ).toBe(false);
    });

    it("flags a branching prefix only when all branches are banned", () => {
      // if both "problem" and "press" are banned, "pr" becomes dead-end.
      const memo = new Map<number, boolean>();
      expect(
        engine.tokenLeadsToBannedWord(
          engine.rootStateId,
          8,
          new Set(["problem", "press"]),
          memo,
        ),
      ).toBe(true);
    });

    it("respects the memo across multiple token checks in one pass", () => {
      const memo = new Map<number, boolean>();
      const banned = new Set(["pioneer"]);
      expect(
        engine.tokenLeadsToBannedWord(engine.rootStateId, 7, banned, memo),
      ).toBe(true);
      expect(memo.size).toBeGreaterThan(0);
      // calling again should reuse memo and still return true
      expect(
        engine.tokenLeadsToBannedWord(engine.rootStateId, 7, banned, memo),
      ).toBe(true);
    });
  });

  describe("getUniqueReachableWord", () => {
    it("returns the single word reachable from a deterministic prefix", () => {
      // wordset where "pi" can only complete to "pioneer"
      const engine = new ConstraintEngine(
        ["pioneer", "problem", "press"],
        [
          { id: 0, text: "pi" },
          { id: 1, text: "pr" },
          { id: 2, text: "oneer" },
        ],
      );
      const piState = engine.getNextState(engine.rootStateId, 0);
      expect(piState).not.toBeNull();
      expect(engine.getUniqueReachableWord(piState as number)).toBe("pioneer");
    });

    it("returns null when a prefix has multiple possible completions", () => {
      const engine = new ConstraintEngine(
        ["pioneer", "problem", "press"],
        [
          { id: 0, text: "pi" },
          { id: 1, text: "pr" },
        ],
      );
      // "pr" state branches to "problem" and "press"
      const prState = engine.getNextState(engine.rootStateId, 1);
      expect(prState).not.toBeNull();
      expect(engine.getUniqueReachableWord(prState as number)).toBeNull();
    });

    it("returns null for the root state (any word reachable)", () => {
      const engine = new ConstraintEngine(
        ["a", "b"],
        [
          { id: 0, text: "a" },
          { id: 1, text: "b" },
        ],
      );
      expect(engine.getUniqueReachableWord(engine.rootStateId)).toBeNull();
    });

    it("returns the word itself for a terminal with no extensions", () => {
      const engine = new ConstraintEngine(
        ["hello"],
        [{ id: 0, text: "hello" }],
      );
      const terminal = engine.getNextState(engine.rootStateId, 0);
      expect(terminal).not.toBeNull();
      expect(engine.getUniqueReachableWord(terminal as number)).toBe("hello");
    });
  });

  it("reports basic precompute stats", () => {
    const engine = new ConstraintEngine(
      ["a", "ab"],
      [
        { id: 0, text: "a" },
        { id: 1, text: "b" },
        { id: 2, text: " ab" },
      ],
    );

    expect(engine.getStats().stateCount).toBe(3);
    expect(engine.getStats().tokenCount).toBe(3);
    expect(engine.getStats().averageValidTokensPerState).toBeCloseTo(4 / 3);
  });
});
