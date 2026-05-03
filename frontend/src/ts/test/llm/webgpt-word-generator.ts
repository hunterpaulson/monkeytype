import { randomElementFromArray } from "../../utils/arrays";
import { Wordset, type FunboxWordsFrequency } from "../wordset";
import { buildSurfaceForms } from "./surface-forms";
import { ConstraintEngine } from "./constraint-engine";
import { acquireWebGptRuntime, releaseWebGptRuntime } from "./webgpt-runtime";
import type { LlmTokenTiming, RuntimeDevice } from "./types";

type WordUpdate = {
  pendingText: string;
  pendingWordStartedInCurrentToken: boolean;
  completedWords: string[];
};

// vLLM / SGLang-shaped sampling params. Every field is optional except
// temperature; defaults make each technique a no-op when omitted so users can
// opt into individual features.
export type SamplingParams = {
  // Softmax temperature. > 1 flattens, < 1 sharpens. 1.0 is the raw model
  // distribution. Must be > 0.
  temperature: number;
  // Keep at most the top-k candidates by logit. Undefined or <= 0 disables
  // top-k (all candidates considered).
  topK?: number;
  // Nucleus sampling: keep smallest set whose cumulative prob >= topP.
  // Not yet implemented — included for API parity with vLLM.
  topP?: number;
  // Min-p sampling: keep candidates with prob >= minP * prob_max. A newer
  // technique (Nguyen 2024, ICLR 2025) that prunes low-prob noise better than
  // top-k when the distribution is flat. Undefined or <= 0 disables it.
  minP?: number;
  // Frequency penalty over recent emitted WORDS (not tokens). For each
  // candidate token that would complete a word present in the recent window,
  // subtract `frequencyPenalty * count` from its logit. Attacks vocab-level
  // attractors that our hard word-rejection filter can't suppress. Undefined
  // or <= 0 disables it. Typical range 0.2-0.7.
  frequencyPenalty?: number;
  // Rolling window (in completed words) over which the frequency penalty
  // counts. Defaults to 50. Larger windows make attractors pay longer.
  frequencyPenaltyWindow?: number;
};

const DEFAULT_SAMPLING_PARAMS: SamplingParams = {
  temperature: 1,
  topK: 0,
  minP: 0.05,
  frequencyPenalty: 2,
  frequencyPenaltyWindow: 100,
};

type WebGptWordGeneratorConfig = {
  contextWindowSize: number;
  initialWords: number;
  bufferMinWords: number;
  bufferTargetWords: number;
  streamingBufferTargetWords: number;
  maxTokensPerFill: number;
  samplingParams: SamplingParams;
  onTokenTiming?: (timing: LlmTokenTiming) => void;
};

const DEFAULT_CONFIG: WebGptWordGeneratorConfig = {
  contextWindowSize: 5,
  initialWords: 1,
  bufferMinWords: 10,
  bufferTargetWords: 100,
  streamingBufferTargetWords: 60,
  maxTokensPerFill: 128,
  samplingParams: DEFAULT_SAMPLING_PARAMS,
};

export class WebGptWordGenerator extends Wordset {
  private readonly config: WebGptWordGeneratorConfig;
  private readonly initPromise: Promise<void>;
  private readonly wordBuffer: string[] = [];

  private engine: ConstraintEngine | null = null;
  private runtime: Awaited<ReturnType<typeof acquireWebGptRuntime>> | null =
    null;
  private tokenizer:
    | Awaited<ReturnType<typeof acquireWebGptRuntime>>["tokenizer"]
    | null = null;
  private decodedTokens: Awaited<
    ReturnType<typeof acquireWebGptRuntime>
  >["decodedTokens"] = [];
  private maxContextWindowSize = 16;
  private currentStateId: number | null = null;
  private allTokenIds: number[] = [];
  private currentWordStartTokenIndex = 0;
  private pendingText = "";
  private requireBoundaryOnNextToken = false;
  private generationPromise: Promise<void> | null = null;
  private requestedBufferSize = 0;
  private disposed = false;
  private initError: Error | null = null;
  private bufferWaiters: Array<() => void> = [];
  private runtimeDevice: RuntimeDevice | null = null;
  private generatedTokenCount = 0;
  // tracks the most recently emitted words to avoid immediate repetition,
  // mirroring monkeytype's default behavior (reject if same as previous 2)
  private recentWords: string[] = [];
  private readonly recentWordsLimit = 2;
  // wider rolling window over emitted words, used by the word-level frequency
  // penalty to suppress vocab-level attractors (e.g. "pioneer" dominating
  // output because GPT-2's prior favors it on short contexts). trimmed to
  // samplingParams.frequencyPenaltyWindow.
  private wordFrequencyHistory: string[] = [];

  constructor(
    words: string[],
    config: Partial<WebGptWordGeneratorConfig> = {},
  ) {
    const surfaceForms = buildSurfaceForms(words);

    if (surfaceForms.length === 0) {
      throw new Error(
        "WebGPT word generator requires at least one usable surface form",
      );
    }

    super(surfaceForms);
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      samplingParams: {
        ...DEFAULT_SAMPLING_PARAMS,
        ...(config.samplingParams ?? {}),
      },
    };
    this.initPromise = this.init(surfaceForms);
  }

  // mode (normal/zipf) is ignored — the LLM's own probability distribution
  // determines word order, not uniform or zipf sampling from the wordset
  override async randomWordAsync(_mode: FunboxWordsFrequency): Promise<string> {
    if (this.disposed) {
      throw new Error("WebGPT word generator was disposed");
    }

    await this.ensureMinBuffer(1);
    return this.randomWord("normal");
  }

  override randomWord(_mode: FunboxWordsFrequency): string {
    if (this.disposed) {
      throw new Error("WebGPT word generator was disposed");
    }

    const word = this.wordBuffer.shift();

    if (word === undefined) {
      throw this.initError ?? new Error("WebGPT word buffer is empty");
    }

    return word;
  }

  override getInitialWordCount(): number | null {
    return this.config.initialWords;
  }

  override getStreamingBufferTarget(): number | null {
    return this.config.streamingBufferTargetWords;
  }

  getRuntimeDevice(): RuntimeDevice | null {
    return this.runtimeDevice;
  }

  getStateCount(): number {
    return this.engine?.getStateCount() ?? 0;
  }

  getMaterializedStateCount(): number {
    return (
      this.engine
        ?.getStateProfiles()
        .filter((stateProfile) => stateProfile.materialized).length ?? 0
    );
  }

  override async dispose(): Promise<void> {
    if (this.runtime !== null) {
      releaseWebGptRuntime(this.runtime);
    }

    this.disposed = true;
    this.wordBuffer.length = 0;
    this.pendingText = "";
    this.generationPromise = null;
    this.requestedBufferSize = 0;
    this.runtime = null;
    this.notifyBufferWaiters();
  }

  async waitForReady(): Promise<void> {
    await this.initPromise;
  }

  async ensureMinBuffer(minWords: number): Promise<void> {
    await this.waitForReady();

    if (this.disposed) {
      return;
    }

    this.requestedBufferSize = Math.max(this.requestedBufferSize, minWords);

    if (this.wordBuffer.length >= this.requestedBufferSize) {
      return;
    }

    this.generationPromise ??= this.fillBuffer();

    while (!this.disposed && this.wordBuffer.length < minWords) {
      const generationPromise = this.generationPromise;

      if (generationPromise === null) {
        this.generationPromise = this.fillBuffer();
        continue;
      }

      await Promise.race([generationPromise, this.waitForBufferChange()]);
    }
  }

  private async init(surfaceForms: string[]): Promise<void> {
    try {
      const runtime = await acquireWebGptRuntime();

      if (this.disposed) {
        releaseWebGptRuntime(runtime);
        return;
      }

      this.runtime = runtime;
      this.tokenizer = runtime.tokenizer;
      this.decodedTokens = runtime.decodedTokens;
      this.maxContextWindowSize = runtime.maxContextWindowSize;
      this.runtimeDevice = runtime.device;
      this.engine = new ConstraintEngine(surfaceForms, this.decodedTokens);

      const seedWord = randomElementFromArray(surfaceForms);
      const promptText = seedWord;
      const currentStateId = this.engine.consumeText(
        this.engine.rootStateId,
        promptText,
      );

      if (currentStateId === null) {
        throw new Error(
          `Failed to seed WebGPT prompt with ${JSON.stringify(promptText)}`,
        );
      }

      this.currentStateId = currentStateId;
      this.allTokenIds = this.tokenizer.encode(promptText);
      this.currentWordStartTokenIndex = this.allTokenIds.length;
      // show the seed word immediately, but force the first sampled token to begin
      // with a space so the visible seed cannot be extended into another word.
      this.wordBuffer.push(seedWord);
      this.requireBoundaryOnNextToken = true;
    } catch (error) {
      this.initError =
        error instanceof Error ? error : new Error(String(error));
      throw this.initError;
    }
  }

  private async fillBuffer(): Promise<void> {
    try {
      while (
        !this.disposed &&
        this.wordBuffer.length < this.requestedBufferSize
      ) {
        await this.generateTokensUntilWords(this.requestedBufferSize);
      }
    } finally {
      this.requestedBufferSize = 0;
      this.generationPromise = null;
      this.notifyBufferWaiters();
    }
  }

  // Runs the core token-by-token generation loop. Each iteration:
  //   1. Builds a bounded context window from recent token IDs
  //   2. Runs a forward pass through the WebGPT model to get logits
  //   3. Masks logits to only tokens valid in the current constraint state
  //      (if this is the first token after the seed word, further restricts
  //       to tokens starting with a space so the seed can't be extended)
  //   4. Samples one token using top-k + temperature
  //   5. Advances the constraint state and appends the token text
  //   6. Splits completed words off the pending text into the word buffer
  //
  // Stops when the buffer reaches targetBufferSize, the generator is disposed,
  // or maxTokensPerFill tokens have been generated (yields back to fillBuffer
  // which re-enters if more words are still needed).
  private async generateTokensUntilWords(
    targetBufferSize: number,
  ): Promise<void> {
    const engine = this.engine;
    const runtime = this.runtime;

    if (engine === null || this.tokenizer === null || runtime === null) {
      throw this.initError ?? new Error("WebGPT generator is not ready");
    }

    for (
      let tokenStep = 0;
      tokenStep < this.config.maxTokensPerFill;
      tokenStep++
    ) {
      if (this.wordBuffer.length >= targetBufferSize || this.disposed) {
        return;
      }

      if (this.currentStateId === null) {
        throw new Error("Current WebGPT state is null");
      }

      const contextWindowSize = Math.min(
        this.config.contextWindowSize,
        this.maxContextWindowSize,
      );
      const contextTokenIds = sliceContextTokenIds(
        this.allTokenIds,
        contextWindowSize,
        this.currentWordStartTokenIndex,
        this.pendingText.length > 0,
      );
      const modelStart = performance.now();
      const logits = await runtime.forward(contextTokenIds);
      const modelMs = performance.now() - modelStart;

      if (this.disposed) {
        return;
      }

      const constraintStart = performance.now();
      const validTokenIds = engine.getValidTokenIds(this.currentStateId);
      const boundaryFiltered = this.requireBoundaryOnNextToken
        ? validTokenIds.filter((tokenId) => {
            // the first sampled token must cross a word boundary because the seed word
            // is already visible in the UI.
            return (this.decodedTokens[tokenId]?.text ?? "").startsWith(" ");
          })
        : validTokenIds;

      // word-level repetition filter: exclude tokens that would either
      // directly complete a recently emitted word OR walk into a dead-end
      // prefix whose only completion is a banned word (e.g. "pi" when
      // "pioneer" is the only pi-word and pioneer is in the banned set).
      // also include the state's current word if it's word-terminal — we may
      // have just emitted a full word in the last token without a trailing
      // space yet, so recentWords doesn't know about it, but the next token
      // could re-emit the same word via a leading-space token.
      // fallback: if every candidate is banned, skip the filter rather than
      // fail — better to repeat than to crash.
      const stateId = this.currentStateId;
      const recentSet = new Set(this.recentWords);
      if (engine.canTerminate(stateId)) {
        recentSet.add(engine.getStatePrefix(stateId));
      }
      const repetitionFiltered =
        recentSet.size === 0
          ? boundaryFiltered
          : (() => {
              const deadEndMemo = new Map<number, boolean>();
              const kept = boundaryFiltered.filter(
                (tokenId) =>
                  !engine.tokenLeadsToBannedWord(
                    stateId,
                    tokenId,
                    recentSet,
                    deadEndMemo,
                  ),
              );
              return kept.length > 0 ? kept : boundaryFiltered;
            })();

      const eligibleTokenIds = repetitionFiltered;

      if (eligibleTokenIds.length === 0) {
        throw new Error(
          `No boundary-respecting WebGPT tokens available from state ${this.currentStateId}`,
        );
      }

      // word-level frequency penalty: for any candidate token that would
      // complete a word present in our rolling frequency history, subtract
      // `frequencyPenalty * count` from its logit. This softens the hard
      // binary filter above with a graduated penalty that keeps attacking
      // attractors as they fall out of the recentWords window.
      const samplingParams = this.config.samplingParams;
      const wordCounts = buildWordCounts(this.wordFrequencyHistory);
      const candidates = eligibleTokenIds.map((tokenId) => {
        const rawScore = logits[tokenId] ?? -Infinity;
        const penalty = computeWordFrequencyPenalty(
          tokenId,
          engine,
          stateId,
          wordCounts,
          samplingParams.frequencyPenalty ?? 0,
        );
        return { tokenId, score: rawScore - penalty };
      });
      const constraintMs = performance.now() - constraintStart;
      const sampleStart = performance.now();
      const nextTokenId = sampleToken(candidates, samplingParams);
      const sampleMs = performance.now() - sampleStart;

      if (nextTokenId === null) {
        throw new Error(
          `Failed to sample WebGPT token from state ${this.currentStateId}`,
        );
      }

      const nextStateId = engine.getNextState(this.currentStateId, nextTokenId);

      if (nextStateId === null) {
        throw new Error(
          `Sampled invalid WebGPT token ${nextTokenId} from state ${this.currentStateId}`,
        );
      }

      if (this.disposed) {
        return;
      }

      this.currentStateId = nextStateId;
      this.allTokenIds.push(nextTokenId);
      this.generatedTokenCount++;
      this.requireBoundaryOnNextToken = false;
      const tokenText = this.decodedTokens[nextTokenId]?.text ?? "";
      const update = consumeTokenTextIntoWords(
        this.pendingText,
        tokenText,
        this.wordBuffer,
      );

      // keep recentWords to the last N emitted, to mirror monkeytype's
      // "reject if same as previous N" rule at generation time
      // also push to the wider wordFrequencyHistory window used by the
      // frequency penalty — a graduated, longer-memory companion to the hard
      // recentWords filter.
      const freqWindow = samplingParams.frequencyPenaltyWindow ?? 0;
      for (const completed of update.completedWords) {
        this.recentWords.push(completed);
        if (this.recentWords.length > this.recentWordsLimit) {
          this.recentWords.shift();
        }
        if (freqWindow > 0) {
          this.wordFrequencyHistory.push(completed);
          while (this.wordFrequencyHistory.length > freqWindow) {
            this.wordFrequencyHistory.shift();
          }
        }
      }

      if (this.wordBuffer.length > 0) {
        this.notifyBufferWaiters();
      }

      this.pendingText = update.pendingText;
      this.currentWordStartTokenIndex = getNextCurrentWordStartTokenIndex(
        this.allTokenIds.length,
        this.currentWordStartTokenIndex,
        this.pendingText,
        update.pendingWordStartedInCurrentToken,
      );
      this.config.onTokenTiming?.({
        tokenIndex: this.generatedTokenCount,
        stateId: this.currentStateId,
        contextLength: contextTokenIds.length,
        validTokenCount: validTokenIds.length,
        bufferSize: this.wordBuffer.length,
        completedWordCount: update.completedWords.length,
        forwardMs: modelMs,
        constraintMs,
        sampleMs,
        totalMs: modelMs + constraintMs + sampleMs,
      });
    }
  }

  private async waitForBufferChange(): Promise<void> {
    return new Promise((resolve) => {
      this.bufferWaiters.push(resolve);
    });
  }

  private notifyBufferWaiters(): void {
    const waiters = this.bufferWaiters;
    this.bufferWaiters = [];

    for (const waiter of waiters) {
      waiter();
    }
  }
}

function consumeTokenTextIntoWords(
  pendingText: string,
  tokenText: string,
  wordBuffer: string[],
): WordUpdate {
  const combinedText = `${pendingText}${tokenText}`;
  const parts = combinedText.split(" ");
  const nextPendingText = parts.pop() ?? "";
  const pendingWordStartedInCurrentToken =
    nextPendingText.length > 0 &&
    (pendingText.length === 0 || tokenText.includes(" "));

  const completedWords: string[] = [];
  for (const word of parts) {
    if (word.length > 0) {
      wordBuffer.push(word);
      completedWords.push(word);
    }
  }

  return {
    pendingText: nextPendingText,
    pendingWordStartedInCurrentToken,
    completedWords,
  };
}

function sliceContextTokenIds(
  allTokenIds: number[],
  contextWindowSize: number,
  currentWordStartTokenIndex: number,
  hasPendingWord: boolean,
): number[] {
  let contextStartIndex = Math.max(0, allTokenIds.length - contextWindowSize);

  if (hasPendingWord && currentWordStartTokenIndex < allTokenIds.length) {
    contextStartIndex = Math.min(contextStartIndex, currentWordStartTokenIndex);
  }

  return allTokenIds.slice(contextStartIndex);
}

function getNextCurrentWordStartTokenIndex(
  totalTokenCount: number,
  currentWordStartTokenIndex: number,
  pendingText: string,
  pendingWordStartedInCurrentToken: boolean,
): number {
  if (pendingText.length === 0) {
    return totalTokenCount;
  }

  if (pendingWordStartedInCurrentToken) {
    return totalTokenCount - 1;
  }

  return currentWordStartTokenIndex;
}

// Builds a word -> count map from a rolling history buffer. O(n) in the buffer
// size; called once per generation step, so fine for small windows (~50).
export function buildWordCounts(history: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const word of history) {
    counts.set(word, (counts.get(word) ?? 0) + 1);
  }
  return counts;
}

// Returns the additive logit penalty for a candidate token, based on how many
// times the word it would commit to has appeared in the frequency history.
// Three cases for the token's next state:
//   - terminal: the word is the state's prefix (direct completion)
//   - non-terminal but deterministically leads to a single word: apply penalty
//     for THAT word. This catches the "pi" -> pioneer trap where the model
//     walks into a prefix whose only completion is an attractor, and the
//     completion-token penalty alone can't save us because the model has
//     already committed.
//   - non-terminal with multiple reachable completions: no penalty; the model
//     hasn't committed yet, so we trust its sampling within that subtree.
export function computeWordFrequencyPenalty(
  tokenId: number,
  engine: ConstraintEngine,
  stateId: number,
  wordCounts: Map<string, number>,
  frequencyPenalty: number,
): number {
  if (frequencyPenalty <= 0 || wordCounts.size === 0) {
    return 0;
  }
  const nextStateId = engine.getNextState(stateId, tokenId);
  if (nextStateId === null) {
    return 0;
  }

  let targetWord: string | null;
  if (engine.canTerminate(nextStateId)) {
    targetWord = engine.getStatePrefix(nextStateId);
  } else {
    targetWord = engine.getUniqueReachableWord(nextStateId);
    if (targetWord === null) {
      return 0;
    }
  }

  const count = wordCounts.get(targetWord) ?? 0;
  return frequencyPenalty * count;
}

// Samples a token from pre-scored candidates using a vLLM-shaped SamplingParams.
// Pipeline: top-k cut → softmax with temperature → min-p filter → categorical.
// Callers pre-apply any logit-space penalties (e.g. frequency penalty) to
// candidate.score before calling this, keeping the sampler itself stateless.
export function sampleToken(
  candidates: Array<{ tokenId: number; score: number }>,
  params: SamplingParams,
  random: () => number = Math.random,
): number | null {
  let filtered = candidates.filter((candidate) =>
    Number.isFinite(candidate.score),
  );
  if (filtered.length === 0) {
    return null;
  }

  filtered.sort((left, right) => right.score - left.score);

  const topK = params.topK;
  if (topK !== undefined && topK > 0 && topK < filtered.length) {
    filtered = filtered.slice(0, topK);
  }

  const temperature = params.temperature > 0 ? params.temperature : 1;
  const maxScore = filtered[0]?.score ?? 0;
  const expScores = filtered.map((candidate) =>
    Math.exp((candidate.score - maxScore) / temperature),
  );
  const expTotal = expScores.reduce((sum, value) => sum + value, 0);
  let probs = expScores.map((value) => value / expTotal);

  const minP = params.minP;
  if (minP !== undefined && minP > 0) {
    const probMax = probs[0] ?? 0; // already sorted desc by score; exp is monotonic
    const threshold = minP * probMax;
    const retainedCandidates: Array<{ tokenId: number; score: number }> = [];
    const retainedProbs: number[] = [];
    for (let index = 0; index < filtered.length; index++) {
      const prob = probs[index] ?? 0;
      const candidate = filtered[index];
      if (prob >= threshold && candidate !== undefined) {
        retainedCandidates.push(candidate);
        retainedProbs.push(prob);
      }
    }
    if (retainedCandidates.length === 0) {
      // should not happen since at least probMax passes its own threshold,
      // but guard to avoid dividing by zero below.
      return filtered[0]?.tokenId ?? null;
    }
    filtered = retainedCandidates;
    const retainedTotal = retainedProbs.reduce((sum, value) => sum + value, 0);
    probs = retainedProbs.map((value) => value / retainedTotal);
  }

  let sample = random();
  for (let index = 0; index < filtered.length; index++) {
    sample -= probs[index] ?? 0;
    if (sample <= 0) {
      return filtered[index]?.tokenId ?? null;
    }
  }

  return filtered.at(-1)?.tokenId ?? null;
}
