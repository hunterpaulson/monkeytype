import { describe, expect, it } from "vitest";
import { ConstraintEngine } from "../../../src/ts/test/llm/constraint-engine";
import {
  buildWordCounts,
  computeWordFrequencyPenalty,
  sampleToken,
  type SamplingParams,
} from "../../../src/ts/test/llm/webgpt-word-generator";

describe("buildWordCounts", () => {
  it("returns empty map for empty history", () => {
    expect(buildWordCounts([]).size).toBe(0);
  });

  it("counts duplicates", () => {
    const counts = buildWordCounts([
      "pioneer",
      "the",
      "pioneer",
      "and",
      "pioneer",
    ]);
    expect(counts.get("pioneer")).toBe(3);
    expect(counts.get("the")).toBe(1);
    expect(counts.get("and")).toBe(1);
  });
});

describe("computeWordFrequencyPenalty", () => {
  const engine = new ConstraintEngine(
    ["pioneer", "the"],
    [
      { id: 0, text: "pioneer" },
      { id: 1, text: "the" },
      { id: 2, text: "pi" },
    ],
  );

  it("returns 0 when the penalty is disabled (0)", () => {
    const counts = buildWordCounts(["pioneer", "pioneer"]);
    expect(
      computeWordFrequencyPenalty(0, engine, engine.rootStateId, counts, 0),
    ).toBe(0);
  });

  it("returns 0 when the history is empty", () => {
    const counts = buildWordCounts([]);
    expect(
      computeWordFrequencyPenalty(0, engine, engine.rootStateId, counts, 0.5),
    ).toBe(0);
  });

  it("applies penalty proportional to count for terminal-landing tokens", () => {
    const counts = buildWordCounts(["pioneer", "the", "pioneer", "pioneer"]);
    // token 0 -> "pioneer" terminal; count[pioneer] = 3; penalty = 0.5 * 3
    expect(
      computeWordFrequencyPenalty(0, engine, engine.rootStateId, counts, 0.5),
    ).toBeCloseTo(1.5);
    // token 1 -> "the" terminal; count[the] = 1; penalty = 0.5 * 1
    expect(
      computeWordFrequencyPenalty(1, engine, engine.rootStateId, counts, 0.5),
    ).toBeCloseTo(0.5);
  });

  it("applies penalty on a non-terminal token that uniquely reaches a hot word", () => {
    // token 2 -> "pi" non-terminal; but "pi" can ONLY extend to "pioneer" in
    // this wordset, so the penalty tracks count[pioneer]. This is the whole
    // point of the prefix-aware frequency penalty — catches attractors that
    // the completion-token penalty alone lets through.
    const counts = buildWordCounts(["pioneer", "pioneer"]);
    expect(
      computeWordFrequencyPenalty(2, engine, engine.rootStateId, counts, 0.5),
    ).toBeCloseTo(1.0);
  });

  it("returns 0 for a non-terminal token that branches (no unique destination)", () => {
    // Build an engine where "pr" can extend to two different words — no
    // unique destination, no penalty.
    const branchingEngine = new ConstraintEngine(
      ["problem", "press"],
      [
        { id: 0, text: "pr" },
        { id: 1, text: "oblem" },
        { id: 2, text: "ess" },
      ],
    );
    const counts = buildWordCounts(["problem", "press", "problem"]);
    expect(
      computeWordFrequencyPenalty(
        0,
        branchingEngine,
        branchingEngine.rootStateId,
        counts,
        0.5,
      ),
    ).toBe(0);
  });

  it("returns 0 when the token is invalid from this state", () => {
    const counts = buildWordCounts(["pioneer"]);
    // token id 999 doesn't exist in the vocab used here
    expect(
      computeWordFrequencyPenalty(999, engine, engine.rootStateId, counts, 0.5),
    ).toBe(0);
  });
});

describe("sampleToken", () => {
  const candidates = [
    { tokenId: 10, score: 5.0 },
    { tokenId: 20, score: 2.0 },
    { tokenId: 30, score: 1.0 },
    { tokenId: 40, score: -1.0 },
  ];

  function constantRandom(value: number): () => number {
    return () => value;
  }

  it("returns null when all candidates are non-finite", () => {
    const result = sampleToken(
      [
        { tokenId: 1, score: Number.NaN },
        { tokenId: 2, score: Number.POSITIVE_INFINITY },
        { tokenId: 3, score: Number.NEGATIVE_INFINITY },
      ],
      { temperature: 1 } satisfies SamplingParams,
      constantRandom(0.5),
    );
    expect(result).toBeNull();
  });

  it("always picks the top-scored token when temperature is very low", () => {
    const params: SamplingParams = { temperature: 0.01 };
    // with near-zero temperature, softmax is nearly a delta at the max
    const result = sampleToken(candidates, params, constantRandom(0.5));
    expect(result).toBe(10);
  });

  it("respects top-k and never samples outside the top k", () => {
    const params: SamplingParams = { temperature: 1, topK: 2 };
    // random near 1 walks to the LAST element of the cumulative
    const result = sampleToken(candidates, params, constantRandom(0.9999));
    // with topK=2, only tokens 10 and 20 are eligible
    expect([10, 20]).toContain(result);
  });

  it("respects min-p by dropping low-probability candidates", () => {
    // without min-p, token 40 (score -1) has some small probability. with
    // minP=0.5 relative to the max prob, it gets dropped entirely.
    const params: SamplingParams = { temperature: 1, minP: 0.5 };
    // force random = 0.999 to land on the LAST retained candidate
    const result = sampleToken(candidates, params, constantRandom(0.999));
    expect(result).not.toBe(40);
    // also token 30 is much lower prob than token 10, so likely filtered
    // at minP=0.5. token 10 or 20 should remain.
    expect([10, 20, 30]).toContain(result);
  });

  it("composes topK + minP + temperature without error", () => {
    const params: SamplingParams = {
      temperature: 0.8,
      topK: 3,
      minP: 0.1,
    };
    const result = sampleToken(candidates, params, constantRandom(0.3));
    expect(result).not.toBeNull();
    expect([10, 20, 30]).toContain(result);
  });

  it("is deterministic given a fixed random source", () => {
    const params: SamplingParams = { temperature: 1 };
    const first = sampleToken(candidates, params, constantRandom(0.2));
    const second = sampleToken(candidates, params, constantRandom(0.2));
    expect(first).toBe(second);
  });

  it("samples the highest-score token when random is 0", () => {
    const params: SamplingParams = { temperature: 1 };
    // with random=0, the first iteration's subtraction makes `sample` <= 0
    // iff probs[0] >= 0, which is always true. so highest-score wins.
    const result = sampleToken(candidates, params, constantRandom(0));
    expect(result).toBe(10);
  });
});
