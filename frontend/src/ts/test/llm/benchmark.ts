/**
 * Benchmarking utilities for constrained decoding performance measurement.
 *
 * Provides timing utilities to measure:
 * - Token string caching time
 * - DFA/state machine construction time
 * - Per-token generation time
 * - Overall generation performance
 */

/**
 * Result of a single timing measurement
 */
export interface TimingResult {
  /** Duration in milliseconds */
  durationMs: number;
  /** Optional label for the measurement */
  label?: string;
}

/**
 * Initialization phase timing breakdown
 */
export interface InitTimingResult {
  /** Time to cache all token strings */
  tokenCacheMs: number;
  /** Time to build the state machine / DFA */
  stateMachineBuildMs: number;
  /** Total initialization time */
  totalMs: number;
}

/**
 * Generation phase timing breakdown
 */
export interface GenerationTimingResult {
  /** Number of tokens generated */
  tokensGenerated: number;
  /** Number of words generated */
  wordsGenerated: number;
  /** Total generation time in ms */
  totalMs: number;
  /** Average time per token in ms */
  msPerToken: number;
  /** Average time per word in ms */
  msPerWord: number;
}

/**
 * State machine statistics
 */
export interface StateStats {
  /** Total number of states in the state machine */
  stateCount: number;
  /** Average number of valid tokens per state */
  avgValidTokensPerState: number;
  /** Min valid tokens for any state */
  minValidTokensPerState: number;
  /** Max valid tokens for any state */
  maxValidTokensPerState: number;
  /** Total number of valid transitions */
  totalValidTransitions: number;
}

/**
 * Complete benchmark result
 */
export interface BenchmarkResult {
  /** Initialization timing */
  initTime: InitTimingResult;
  /** Generation timing (if generation was run) */
  generationTime?: GenerationTimingResult;
  /** State machine statistics */
  stateStats: StateStats;
  /** Wordset size used */
  wordsetSize: number;
  /** Vocabulary size */
  vocabSize: number;
}

/**
 * Simple high-resolution timer for benchmarking.
 * Uses performance.now() in browser, process.hrtime() in Node.js
 */
export class Timer {
  private startTime: number = 0;
  private endTime: number = 0;
  private running: boolean = false;

  /**
   * Start the timer
   */
  start(): void {
    this.startTime = this.now();
    this.running = true;
  }

  /**
   * Stop the timer and return elapsed time in milliseconds
   */
  stop(): number {
    if (!this.running) {
      return 0;
    }
    this.endTime = this.now();
    this.running = false;
    return this.endTime - this.startTime;
  }

  /**
   * Get elapsed time without stopping
   */
  elapsed(): number {
    if (this.running) {
      return this.now() - this.startTime;
    }
    return this.endTime - this.startTime;
  }

  /**
   * Reset the timer
   */
  reset(): void {
    this.startTime = 0;
    this.endTime = 0;
    this.running = false;
  }

  /**
   * Get current time in milliseconds
   */
  private now(): number {
    if (typeof performance !== "undefined" && performance.now) {
      return performance.now();
    }
    // Node.js fallback
    if (typeof process !== "undefined" && process.hrtime) {
      const hr = process.hrtime();
      return hr[0] * 1000 + hr[1] / 1e6;
    }
    return Date.now();
  }
}

/**
 * Benchmark context for collecting timing data during initialization and generation.
 */
export class BenchmarkContext {
  private initTimer = new Timer();
  private tokenCacheTimer = new Timer();
  private stateMachineBuildTimer = new Timer();
  private generationTimer = new Timer();

  private tokenCacheMs = 0;
  private stateMachineBuildMs = 0;
  private initTotalMs = 0;

  private tokensGenerated = 0;
  private wordsGenerated = 0;
  private generationTotalMs = 0;

  private stateStats: StateStats = {
    stateCount: 0,
    avgValidTokensPerState: 0,
    minValidTokensPerState: Infinity,
    maxValidTokensPerState: 0,
    totalValidTransitions: 0,
  };

  private wordsetSize = 0;
  private vocabSize = 0;

  // ---- Initialization timing ----

  startInit(): void {
    this.initTimer.start();
  }

  endInit(): void {
    this.initTotalMs = this.initTimer.stop();
  }

  startTokenCache(): void {
    this.tokenCacheTimer.start();
  }

  endTokenCache(): void {
    this.tokenCacheMs = this.tokenCacheTimer.stop();
  }

  startStateMachineBuild(): void {
    this.stateMachineBuildTimer.start();
  }

  endStateMachineBuild(): void {
    this.stateMachineBuildMs = this.stateMachineBuildTimer.stop();
  }

  // ---- Generation timing ----

  startGeneration(): void {
    this.generationTimer.start();
    this.tokensGenerated = 0;
    this.wordsGenerated = 0;
  }

  recordToken(): void {
    this.tokensGenerated++;
  }

  recordWord(): void {
    this.wordsGenerated++;
  }

  endGeneration(): void {
    this.generationTotalMs = this.generationTimer.stop();
  }

  // ---- State stats ----

  setStateStats(stats: StateStats): void {
    this.stateStats = stats;
  }

  setWordsetSize(size: number): void {
    this.wordsetSize = size;
  }

  setVocabSize(size: number): void {
    this.vocabSize = size;
  }

  // ---- Results ----

  getInitTiming(): InitTimingResult {
    return {
      tokenCacheMs: this.tokenCacheMs,
      stateMachineBuildMs: this.stateMachineBuildMs,
      totalMs: this.initTotalMs,
    };
  }

  getGenerationTiming(): GenerationTimingResult | undefined {
    if (this.generationTotalMs === 0) {
      return undefined;
    }
    return {
      tokensGenerated: this.tokensGenerated,
      wordsGenerated: this.wordsGenerated,
      totalMs: this.generationTotalMs,
      msPerToken:
        this.tokensGenerated > 0
          ? this.generationTotalMs / this.tokensGenerated
          : 0,
      msPerWord:
        this.wordsGenerated > 0
          ? this.generationTotalMs / this.wordsGenerated
          : 0,
    };
  }

  getResult(): BenchmarkResult {
    return {
      initTime: this.getInitTiming(),
      generationTime: this.getGenerationTiming(),
      stateStats: this.stateStats,
      wordsetSize: this.wordsetSize,
      vocabSize: this.vocabSize,
    };
  }

  /**
   * Format benchmark results as a human-readable string
   */
  formatResults(): string {
    const result = this.getResult();
    const lines: string[] = [];

    lines.push("=".repeat(60));
    lines.push("BENCHMARK RESULTS");
    lines.push("=".repeat(60));

    lines.push("\n--- Configuration ---");
    lines.push(`Wordset size: ${result.wordsetSize} words`);
    lines.push(`Vocabulary size: ${result.vocabSize} tokens`);

    lines.push("\n--- Initialization ---");
    lines.push(`Token cache:        ${result.initTime.tokenCacheMs.toFixed(2)} ms`);
    lines.push(`State machine build: ${result.initTime.stateMachineBuildMs.toFixed(2)} ms`);
    lines.push(`Total init time:    ${result.initTime.totalMs.toFixed(2)} ms`);

    lines.push("\n--- State Machine Stats ---");
    lines.push(`States:              ${result.stateStats.stateCount}`);
    lines.push(`Avg valid tokens:    ${result.stateStats.avgValidTokensPerState.toFixed(1)}`);
    lines.push(`Min valid tokens:    ${result.stateStats.minValidTokensPerState}`);
    lines.push(`Max valid tokens:    ${result.stateStats.maxValidTokensPerState}`);
    lines.push(`Total transitions:   ${result.stateStats.totalValidTransitions}`);

    if (result.generationTime) {
      lines.push("\n--- Generation ---");
      lines.push(`Tokens generated:    ${result.generationTime.tokensGenerated}`);
      lines.push(`Words generated:     ${result.generationTime.wordsGenerated}`);
      lines.push(`Total time:          ${result.generationTime.totalMs.toFixed(2)} ms`);
      lines.push(`Time per token:      ${result.generationTime.msPerToken.toFixed(3)} ms`);
      lines.push(`Time per word:       ${result.generationTime.msPerWord.toFixed(3)} ms`);
    }

    lines.push("\n" + "=".repeat(60));

    return lines.join("\n");
  }
}

/**
 * Helper to time a synchronous function
 */
export function timeSync<T>(fn: () => T): { result: T; durationMs: number } {
  const timer = new Timer();
  timer.start();
  const result = fn();
  const durationMs = timer.stop();
  return { result, durationMs };
}

/**
 * Helper to time an async function
 */
export async function timeAsync<T>(
  fn: () => Promise<T>
): Promise<{ result: T; durationMs: number }> {
  const timer = new Timer();
  timer.start();
  const result = await fn();
  const durationMs = timer.stop();
  return { result, durationMs };
}


