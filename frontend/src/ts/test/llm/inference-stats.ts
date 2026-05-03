import type { LlmTokenTiming } from "./types";

type Listener = () => void;

export type LlmInferenceStatsSnapshot = {
  tokenCount: number;
  completedWordCount: number;
  elapsedMs: number;
  tokensPerSecond: number | null;
  wordsPerMinute: number | null;
  tokensPerCompletedWord: number | null;
  rollingTokensPerSecond: number | null;
  rollingWordsPerMinute: number | null;
  rollingTokensPerCompletedWord: number | null;
};

const listeners = new Set<Listener>();
const DEBUG_INFERENCE_STATS =
  (typeof localStorage !== "undefined" &&
    localStorage.getItem("typegptDebugInferenceStats") === "true") ||
  (typeof location !== "undefined" &&
    new URLSearchParams(location.search).get("debugInferenceStats") === "true");
const ROLLING_WINDOW_SIZE = 128;

let firstTokenAt: number | null = null;
let lastTokenAt: number | null = null;
let tokenCount = 0;
let completedWordCount = 0;
let lastDebugLogAt = 0;
const rollingEvents: Array<{
  at: number;
  completedWordCount: number;
}> = [];

export function resetLlmInferenceStats(): void {
  firstTokenAt = null;
  lastTokenAt = null;
  tokenCount = 0;
  completedWordCount = 0;
  lastDebugLogAt = 0;
  rollingEvents.length = 0;
  notify();
}

export function recordLlmTokenTiming(timing: LlmTokenTiming): void {
  const now = performance.now();

  firstTokenAt ??= now;
  lastTokenAt = now;
  tokenCount++;
  completedWordCount += timing.completedWordCount;
  rollingEvents.push({
    at: now,
    completedWordCount: timing.completedWordCount,
  });
  while (rollingEvents.length > ROLLING_WINDOW_SIZE) {
    rollingEvents.shift();
  }

  debugLogInferenceStats(now, timing);

  notify();
}

export function getLlmInferenceStatsSnapshot(): LlmInferenceStatsSnapshot {
  const elapsedMs =
    firstTokenAt !== null && lastTokenAt !== null
      ? Math.max(lastTokenAt - firstTokenAt, 0)
      : 0;
  const elapsedSeconds = elapsedMs / 1000;
  const elapsedMinutes = elapsedMs / 60000;
  const rollingElapsedMs =
    rollingEvents.length > 1
      ? Math.max(
          (rollingEvents.at(-1)?.at ?? 0) - (rollingEvents[0]?.at ?? 0),
          0,
        )
      : 0;
  const rollingCompletedWordCount = rollingEvents.reduce(
    (sum, event) => sum + event.completedWordCount,
    0,
  );

  return {
    tokenCount,
    completedWordCount,
    elapsedMs,
    tokensPerSecond:
      tokenCount > 1 && elapsedSeconds > 0 ? tokenCount / elapsedSeconds : null,
    wordsPerMinute:
      completedWordCount > 0 && elapsedMinutes > 0
        ? completedWordCount / elapsedMinutes
        : null,
    tokensPerCompletedWord:
      completedWordCount > 0 ? tokenCount / completedWordCount : null,
    rollingTokensPerSecond:
      rollingEvents.length > 1 && rollingElapsedMs > 0
        ? rollingEvents.length / (rollingElapsedMs / 1000)
        : null,
    rollingWordsPerMinute:
      rollingCompletedWordCount > 0 && rollingElapsedMs > 0
        ? rollingCompletedWordCount / (rollingElapsedMs / 60000)
        : null,
    rollingTokensPerCompletedWord:
      rollingCompletedWordCount > 0
        ? rollingEvents.length / rollingCompletedWordCount
        : null,
  };
}

export function subscribeLlmInferenceStats(listener: Listener): () => void {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
}

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

function debugLogInferenceStats(now: number, timing: LlmTokenTiming): void {
  if (!DEBUG_INFERENCE_STATS || now - lastDebugLogAt < 1000) {
    return;
  }

  lastDebugLogAt = now;
  const snapshot = getLlmInferenceStatsSnapshot();
  console.log(
    "TYPEGPT_INFERENCE_STATS",
    JSON.stringify({
      latestToken: timing.tokenIndex,
      latestCompletedWords: timing.completedWordCount,
      bufferSize: timing.bufferSize,
      totalTokens: snapshot.tokenCount,
      totalCompletedWords: snapshot.completedWordCount,
      elapsedSeconds: (snapshot.elapsedMs / 1000).toFixed(1),
      avgTokPerSecond: snapshot.tokensPerSecond?.toFixed(1) ?? "warming",
      avgGeneratedWpm: snapshot.wordsPerMinute?.toFixed(0) ?? "warming",
      avgTokensPerWord:
        snapshot.tokensPerCompletedWord?.toFixed(2) ?? "warming",
      rollingTokPerSecond:
        snapshot.rollingTokensPerSecond?.toFixed(1) ?? "warming",
      rollingGeneratedWpm:
        snapshot.rollingWordsPerMinute?.toFixed(0) ?? "warming",
      rollingTokensPerWord:
        snapshot.rollingTokensPerCompletedWord?.toFixed(2) ?? "warming",
      forwardMs: timing.forwardMs.toFixed(1),
      constraintMs: timing.constraintMs.toFixed(1),
      sampleMs: timing.sampleMs.toFixed(1),
    }),
  );
}
