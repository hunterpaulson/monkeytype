import assert from "node:assert/strict";
import { Gpt2ConstrainedLogitsProcessor } from "../gpt2-constrained-logits-processor";

type FakeTokConfig = {
  vocab: Record<number, string>;
};

function makeFakeTokenizer(config: FakeTokConfig) {
  return {
    decode(tokens: Array<number | bigint>) {
      return tokens
        .map((t) => {
          const id = typeof t === "bigint" ? Number(t) : t;
          return config.vocab[id] ?? "";
        })
        .join("");
    },
  };
}

function buildProcessor() {
  const vocab: Record<number, string> = {
    0: " ",
    1: "h",
    2: "e",
    3: "l",
    4: "o",
    5: "w",
    6: "r",
    7: "d",
    8: "!",
  };
  const tokenizer = makeFakeTokenizer({ vocab });
  const wordset = ["hello", "world"];
  const processor = new Gpt2ConstrainedLogitsProcessor(
    wordset,
    tokenizer,
    Object.keys(vocab).length,
  );
  return { processor, tokenizer, vocabSize: Object.keys(vocab).length };
}

function maskWithProcessor(
  processor: Gpt2ConstrainedLogitsProcessor,
  inputIds: number[],
  vocabSize: number,
) {
  const logits = new Float32Array(vocabSize).fill(0);
  const tensor = { data: logits, dims: [1, vocabSize] };
  processor.process([inputIds.map((v) => BigInt(v))], tensor);
  return logits;
}

function test(name: string, fn: () => void) {
  try {
    fn();
    // eslint-disable-next-line no-console
    console.log(`✓ ${name}`);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(`✗ ${name}`);
    throw err;
  }
}

test("allows starting prefixes for words", () => {
  const { processor, vocabSize } = buildProcessor();
  const logits = maskWithProcessor(processor, [0], vocabSize); // starts with space

  assert.notEqual(logits[1], -Infinity);
  assert.notEqual(logits[5], -Infinity);
  assert.equal(logits[8], -Infinity);
});

test("advances through hello and resets at boundary", () => {
  const { processor, vocabSize } = buildProcessor();
  // " hello " as tokens: space h e l l o space
  const seq = [0, 1, 2, 3, 3, 4, 0].map((v) => BigInt(v));
  const logits = new Float32Array(vocabSize).fill(0);
  processor.process([seq], { data: logits, dims: [1, vocabSize] });

  // Should allow starting world with "w"
  assert.notEqual(logits[5], -Infinity);
});

test("rejects invalid continuation mid-word", () => {
  const { processor, vocabSize } = buildProcessor();
  const logits = maskWithProcessor(processor, [0, 1, 2], vocabSize);

  assert.equal(logits[8], -Infinity);
});

