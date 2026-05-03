import type { LlmTokenTiming, RuntimeDevice } from "./types";
import { clearWebGptRuntimeCache, loadWebGptRuntime } from "./webgpt-runtime";
import {
  WebGptWordGenerator,
  type SamplingParams,
} from "./webgpt-word-generator";

export type BrowserLlmBenchmarkOptions = {
  languageFile?: string;
  wordsToConsume?: number;
  contextWindowSize?: number;
  minP?: number;
  frequencyPenalty?: number;
  frequencyPenaltyWindow?: number;
  topK?: number;
  temperature?: number;
  clearRuntimeCache?: boolean;
};

type MemorySnapshot = {
  jsHeapSizeLimit?: number;
  totalJSHeapSize?: number;
  usedJSHeapSize?: number;
  uaBytes?: number;
  uaBreakdown?: Array<{ bytes: number; types: string[] }>;
};

type ResourceSnapshot = {
  name: string;
  duration: number;
  transferSize: number;
  encodedBodySize: number;
  decodedBodySize: number;
  initiatorType: string;
};

export type BrowserLlmBenchmarkResult = {
  modelId: string;
  languageFile: string;
  wordsToConsume: number;
  contextWindowSize: number;
  samplingParams: SamplingParams;
  runtimeDevice: RuntimeDevice | null;
  runtimeLoadMs: number;
  generatorReadyMs: number;
  firstWordMs: number;
  totalWordConsumptionMs: number;
  wordsPerSecond: number;
  wordLatenciesMs: number[];
  tokenTimings: LlmTokenTiming[];
  tokenTimingSummary: {
    avgTotalMs: number;
    p95TotalMs: number;
    avgForwardMs: number;
    avgConstraintMs: number;
    avgSampleMs: number;
  };
  stateSummary: {
    stateCount: number;
    materializedStateCount: number;
  };
  qualityMetrics: LlmBenchmarkQualityMetrics;
  memoryBefore: MemorySnapshot;
  memoryAfter: MemorySnapshot;
  resources: ResourceSnapshot[];
  words: string[];
  sampleText: string;
};

export type LlmBenchmarkQualityMetrics = {
  immediateRepeatCount: number;
  immediateRepeatRate: number;
  repeatWithinLast10Count: number;
  repeatWithinLast10Rate: number;
  uniqueWordCount: number;
  uniqueWordRate: number;
  averageWordLength: number;
  shortWordRate: number;
  mediumWordRate: number;
  longWordRate: number;
  averageTokensPerCompletedWord: number;
  completedWordsPerToken: number;
  generatedWordsPerMinute: number;
  generatedTokensPerSecond: number;
  topWords: Array<{ word: string; count: number }>;
  topBigrams: Array<{ bigram: string; count: number }>;
};

export async function runLlmBrowserBenchmark(
  options: BrowserLlmBenchmarkOptions = {},
): Promise<BrowserLlmBenchmarkResult> {
  const languageFile = options.languageFile ?? "english_5k.json";
  const wordsToConsume = options.wordsToConsume ?? 100;
  const contextWindowSize = options.contextWindowSize ?? 5;
  const samplingParams: SamplingParams = {
    temperature: options.temperature ?? 1,
    topK: options.topK ?? 0,
    minP: options.minP ?? 0.1,
    frequencyPenalty: options.frequencyPenalty ?? 2,
    frequencyPenaltyWindow: options.frequencyPenaltyWindow ?? 100,
  };

  if (options.clearRuntimeCache) {
    clearWebGptRuntimeCache();
  }

  const resourceStartTime = performance.now();
  const memoryBefore = await captureMemorySnapshot();

  const runtimeStart = performance.now();
  const loadedRuntime = await loadWebGptRuntime();
  const runtimeLoadMs = performance.now() - runtimeStart;

  const words = await loadLanguageWords(languageFile);
  const tokenTimings: LlmTokenTiming[] = [];
  const generatorReadyStart = performance.now();
  const generator = new WebGptWordGenerator(words, {
    contextWindowSize,
    samplingParams,
    onTokenTiming(timing) {
      tokenTimings.push(timing);
    },
  });

  await generator.waitForReady();
  await generator.ensureMinBuffer(generator.getInitialWordCount() ?? 1);
  const generatorReadyMs = performance.now() - generatorReadyStart;

  const consumedWords: string[] = [];
  const wordLatenciesMs: number[] = [];
  const firstWordStart = performance.now();

  for (let index = 0; index < wordsToConsume; index++) {
    const wordStart = performance.now();
    const word = await generator.randomWordAsync("normal");
    const wordLatencyMs = performance.now() - wordStart;

    if (index === 0) {
      wordLatenciesMs.push(performance.now() - firstWordStart);
    } else {
      wordLatenciesMs.push(wordLatencyMs);
    }

    consumedWords.push(word);
  }

  const totalWordConsumptionMs = wordLatenciesMs.reduce(
    (sum, value) => sum + value,
    0,
  );
  const memoryAfter = await captureMemorySnapshot();
  const resources = collectResourceSnapshots(resourceStartTime);

  const result: BrowserLlmBenchmarkResult = {
    modelId: loadedRuntime.modelId,
    languageFile,
    wordsToConsume,
    contextWindowSize,
    samplingParams,
    runtimeDevice: generator.getRuntimeDevice() ?? loadedRuntime.device,
    runtimeLoadMs,
    generatorReadyMs,
    firstWordMs: wordLatenciesMs[0] ?? 0,
    totalWordConsumptionMs,
    wordsPerSecond:
      totalWordConsumptionMs === 0
        ? 0
        : (wordsToConsume / totalWordConsumptionMs) * 1000,
    wordLatenciesMs,
    tokenTimings,
    tokenTimingSummary: {
      avgTotalMs: average(tokenTimings.map((timing) => timing.totalMs)),
      p95TotalMs: percentile(
        tokenTimings.map((timing) => timing.totalMs),
        0.95,
      ),
      avgForwardMs: average(tokenTimings.map((timing) => timing.forwardMs)),
      avgConstraintMs: average(
        tokenTimings.map((timing) => timing.constraintMs),
      ),
      avgSampleMs: average(tokenTimings.map((timing) => timing.sampleMs)),
    },
    stateSummary: {
      stateCount: generator.getStateCount(),
      materializedStateCount: generator.getMaterializedStateCount(),
    },
    qualityMetrics: calculateQualityMetrics(consumedWords, tokenTimings),
    memoryBefore,
    memoryAfter,
    resources,
    words: consumedWords,
    sampleText: consumedWords.join(" "),
  };

  await generator.dispose();
  return result;
}

function calculateQualityMetrics(
  words: string[],
  tokenTimings: LlmTokenTiming[],
): LlmBenchmarkQualityMetrics {
  let immediateRepeatCount = 0;
  let repeatWithinLast10Count = 0;
  const wordCounts = new Map<string, number>();
  const bigramCounts = new Map<string, number>();

  for (let index = 0; index < words.length; index++) {
    const word = normalizeWord(words[index] ?? "");
    wordCounts.set(word, (wordCounts.get(word) ?? 0) + 1);

    if (index > 0) {
      const previous = normalizeWord(words[index - 1] ?? "");
      if (word === previous) {
        immediateRepeatCount++;
      }
      const bigram = `${previous} ${word}`;
      bigramCounts.set(bigram, (bigramCounts.get(bigram) ?? 0) + 1);
    }

    const recentWindowStart = Math.max(0, index - 10);
    for (
      let recentIndex = recentWindowStart;
      recentIndex < index;
      recentIndex++
    ) {
      if (normalizeWord(words[recentIndex] ?? "") === word) {
        repeatWithinLast10Count++;
        break;
      }
    }
  }

  const generatedTokenCount = tokenTimings.length;
  const completedWordCount = tokenTimings.reduce(
    (sum, timing) => sum + timing.completedWordCount,
    0,
  );
  const totalGenerationMs = tokenTimings.reduce(
    (sum, timing) => sum + timing.totalMs,
    0,
  );

  return {
    immediateRepeatCount,
    immediateRepeatRate: rate(
      immediateRepeatCount,
      Math.max(words.length - 1, 0),
    ),
    repeatWithinLast10Count,
    repeatWithinLast10Rate: rate(repeatWithinLast10Count, words.length),
    uniqueWordCount: wordCounts.size,
    uniqueWordRate: rate(wordCounts.size, words.length),
    averageWordLength: average(words.map((word) => word.length)),
    shortWordRate: rate(
      words.filter((word) => word.length <= 3).length,
      words.length,
    ),
    mediumWordRate: rate(
      words.filter((word) => word.length >= 4 && word.length <= 7).length,
      words.length,
    ),
    longWordRate: rate(
      words.filter((word) => word.length >= 8).length,
      words.length,
    ),
    averageTokensPerCompletedWord: rate(
      generatedTokenCount,
      completedWordCount,
    ),
    completedWordsPerToken: rate(completedWordCount, generatedTokenCount),
    generatedWordsPerMinute:
      totalGenerationMs > 0
        ? (completedWordCount / totalGenerationMs) * 60000
        : 0,
    generatedTokensPerSecond:
      totalGenerationMs > 0
        ? (generatedTokenCount / totalGenerationMs) * 1000
        : 0,
    topWords: topEntries(wordCounts, 10).map(([word, count]) => ({
      word,
      count,
    })),
    topBigrams: topEntries(bigramCounts, 10).map(([bigram, count]) => ({
      bigram,
      count,
    })),
  };
}

function normalizeWord(word: string): string {
  return word.toLowerCase();
}

function topEntries(
  map: Map<string, number>,
  limit: number,
): Array<[string, number]> {
  return Array.from(map.entries())
    .sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    )
    .slice(0, limit);
}

function rate(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

async function loadLanguageWords(languageFile: string): Promise<string[]> {
  const response = await fetch(`/languages/${languageFile}`);

  if (!response.ok) {
    throw new Error(
      `Failed to fetch language file ${languageFile}: ${response.status}`,
    );
  }

  const parsed = (await response.json()) as { words?: string[] };

  if (!parsed.words) {
    throw new Error(`Language file missing words: ${languageFile}`);
  }

  return parsed.words;
}

async function captureMemorySnapshot(): Promise<MemorySnapshot> {
  const memorySnapshot: MemorySnapshot = {};

  if ("memory" in performance) {
    const memory = (
      performance as Performance & {
        memory?: {
          jsHeapSizeLimit: number;
          totalJSHeapSize: number;
          usedJSHeapSize: number;
        };
      }
    ).memory;

    if (memory) {
      memorySnapshot.jsHeapSizeLimit = memory.jsHeapSizeLimit;
      memorySnapshot.totalJSHeapSize = memory.totalJSHeapSize;
      memorySnapshot.usedJSHeapSize = memory.usedJSHeapSize;
    }
  }

  if ("measureUserAgentSpecificMemory" in performance) {
    try {
      const detailedMemory = await (
        performance as Performance & {
          measureUserAgentSpecificMemory?: () => Promise<{
            bytes: number;
            breakdown: Array<{ bytes: number; types: string[] }>;
          }>;
        }
      ).measureUserAgentSpecificMemory?.();

      if (detailedMemory) {
        memorySnapshot.uaBytes = detailedMemory.bytes;
        memorySnapshot.uaBreakdown = detailedMemory.breakdown;
      }
    } catch {
      // ignore browser memory API failures
    }
  }

  return memorySnapshot;
}

function collectResourceSnapshots(startTime: number): ResourceSnapshot[] {
  return performance
    .getEntriesByType("resource")
    .filter((entry): entry is PerformanceResourceTiming => {
      return (
        entry instanceof PerformanceResourceTiming &&
        entry.startTime >= startTime &&
        (entry.name.includes("huggingface.co") ||
          entry.name.includes("githubusercontent.com") ||
          entry.name.includes("jsdelivr.net") ||
          entry.name.includes("gpt_tokens.json") ||
          entry.name.includes("vocab.bpe") ||
          entry.name.includes("/languages/"))
      );
    })
    .map((entry) => ({
      name: entry.name,
      duration: entry.duration,
      transferSize: entry.transferSize,
      encodedBodySize: entry.encodedBodySize,
      decodedBodySize: entry.decodedBodySize,
      initiatorType: entry.initiatorType,
    }));
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: number[], ratio: number): number {
  if (values.length === 0) {
    return 0;
  }

  const sortedValues = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.floor(sortedValues.length * ratio)),
  );

  return sortedValues[index] ?? 0;
}
