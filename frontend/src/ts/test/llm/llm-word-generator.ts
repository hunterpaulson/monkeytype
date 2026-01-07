import { Wordset } from "../wordset";
import {
  pipeline,
  TextGenerationPipeline,
  env,
} from "@huggingface/transformers";
import { WordsetConstrainedLogitsProcessor } from "./constrained-logits-processor";

// Configure transformers.js to use browser cache
// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
env.useBrowserCache = true;
// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
env.allowLocalModels = false;

/**
 * Configuration for the LLM word generator
 */
export type LLMConfig = {
  /** Model ID from Hugging Face */
  modelId: string;
  /** Maximum number of tokens in context (to prevent going off rails) */
  maxContextTokens: number;
  /** Minimum words to keep in buffer before triggering generation */
  bufferMinWords: number;
  /** Target number of words to generate per batch */
  batchSize: number;
  /** Temperature for sampling (higher = more random) */
  temperature: number;
  /** Top-p (nucleus) sampling threshold */
  topP: number;
  /** Whether to use constrained decoding (wordset-based) */
  useConstrainedDecoding: boolean;
};

const DEFAULT_CONFIG: LLMConfig = {
  // DistilGPT-2: ~82M params, base model (distilled from GPT-2, no RLHF)
  // This is a true base model that just does next-token prediction
  modelId: "Xenova/distilgpt2",
  maxContextTokens: 10, // Small context to prevent going off rails
  bufferMinWords: 5,
  batchSize: 15,
  temperature: 0.8, // Slightly lower for more consistent output
  topP: 0.9, // Slightly tighter nucleus
  useConstrainedDecoding: false, // Full wordset constraint via trie-based logits masking
};

/**
 * Enum for LLM engine initialization states
 */
export enum LLMState {
  UNINITIALIZED = "uninitialized",
  LOADING = "loading",
  READY = "ready",
  ERROR = "error",
}

/**
 * Error thrown when LLM operations fail
 */
export class LLMError extends Error {
  public readonly originalCause?: unknown;

  constructor(message: string, originalCause?: unknown) {
    super(message);
    this.name = "LLMError";
    this.originalCause = originalCause;
  }
}

/**
 * Progress callback type for model loading
 */
export type ProgressCallback = (progress: {
  status: string;
  progress?: number;
  file?: string;
}) => void;

/**
 * LLMWordGenerator extends Wordset to provide LLM-generated word sequences.
 *
 * Uses transformers.js with DistilGPT-2 (~82M params) - a true BASE model
 * that's just pretrained for next-token prediction (no RLHF, no instruction tuning).
 *
 * Architecture:
 * ```
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                      LLMWordGenerator                           │
 * │                                                                 │
 * │  randomWord() ◄── wordBuffer[] ◄── generateBatch() ◄── LLM    │
 * │       │                                    │                    │
 * │       └── (sync, returns immediately)      └── (async, batched) │
 * │                                                                 │
 * │  Constrained Decoding:                                          │
 * │  - LogitsProcessor masks invalid tokens at each step           │
 * │  - Trie-based wordset lookup for valid continuations           │
 * │  - Guarantees output words are from the wordset                │
 * └─────────────────────────────────────────────────────────────────┘
 * ```
 */
export class LLMWordGenerator extends Wordset {
  private config: LLMConfig;
  private wordBuffer: string[] = [];
  private state: LLMState = LLMState.UNINITIALIZED;
  private stateError: Error | null = null;
  private isGenerating: boolean = false;
  private generator: TextGenerationPipeline | null = null;
  private initPromise: Promise<void> | null = null;

  // Constrained decoding components
  private constrainedProcessor: WordsetConstrainedLogitsProcessor | null =
    null;
  private sourceWords: string[];

  // Progress callback for UI updates during model loading
  public onProgress: ProgressCallback | null = null;

  constructor(words: string[], config: Partial<LLMConfig> = {}) {
    super(words);
    this.sourceWords = words;
    this.config = { ...DEFAULT_CONFIG, ...config };

    // LLM can generate unbounded words
    this.length = Infinity;

    // Start loading the LLM engine asynchronously
    this.initPromise = this.initEngine();
  }

  /**
   * Initialize the transformers.js pipeline.
   * This runs asynchronously - call waitForReady() to ensure it's loaded.
   */
  private async initEngine(): Promise<void> {
    this.state = LLMState.LOADING;

    try {
      console.log(`[LLM Funbox] Loading model: ${this.config.modelId}`);

      // Track loading stages for cleaner logging
      let lastStatus = "";
      let lastFile = "";

      // Create text-generation pipeline with progress callback
      const progressCallback = (progress: {
        status: string;
        progress?: number;
        file?: string;
      }): void => {
        // Only log meaningful progress updates (reduce noise)
        const isNewFile = progress.file !== lastFile;
        const isNewStatus = progress.status !== lastStatus;
        const isComplete =
          progress.progress !== undefined && progress.progress >= 99;

        if (isNewStatus || (isNewFile && progress.file !== undefined)) {
          lastStatus = progress.status;
          lastFile = progress.file ?? "";

          // Simplified logging
          if (progress.status === "ready") {
            console.log(`[LLM Funbox] Model ready`);
          } else if (progress.file !== undefined && isComplete) {
            console.log(`[LLM Funbox] Loaded ${progress.file}`);
          }
        }

        if (this.onProgress) {
          this.onProgress(progress);
        }
      };

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
      const gen = await pipeline("text-generation", this.config.modelId, {
        progress_callback: progressCallback,
      });
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      this.generator = gen as TextGenerationPipeline;

      // Initialize constrained decoding processors if enabled
      await this.initConstrainedDecoding();

      this.state = LLMState.READY;
      console.log("[LLM Funbox] Engine ready");

      // Pre-fill the buffer
      await this.refillBuffer();
    } catch (error) {
      console.error("[LLM Funbox] Failed to initialize engine:", error);
      this.state = LLMState.ERROR;
      this.stateError =
        error instanceof Error ? error : new Error(String(error));
      throw new LLMError("Failed to initialize LLM engine", error);
    }
  }

  /**
   * Initialize constrained decoding processors.
   */
  private async initConstrainedDecoding(): Promise<void> {
    // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
    if (!this.generator) return;

    // Get tokenizer from the pipeline
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    const tokenizer = (this.generator as any).tokenizer;
    // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
    if (!tokenizer) {
      console.warn(
        "[LLM Funbox] Could not access tokenizer for constrained decoding"
      );
      return;
    }

    // Get vocab size from model config
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    const vocabSize =
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
      (this.generator as any).model?.config?.vocab_size ?? 50257; // GPT-2 default

    if (this.config.useConstrainedDecoding) {
      console.log(
        "[LLM Funbox] Initializing wordset-constrained decoding..."
      );
      this.constrainedProcessor = new WordsetConstrainedLogitsProcessor(
        this.sourceWords,
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        tokenizer,
        vocabSize as number,
        { debug: false }
      );
    }
  }

  /**
   * Wait for the engine to be ready.
   * Throws if initialization failed.
   */
  public async waitForReady(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise;
    }
    if (this.state === LLMState.ERROR) {
      throw this.stateError ?? new LLMError("LLM engine failed to initialize");
    }
  }

  /**
   * Build the prompt for text generation.
   * We use an empty prompt to simplify constrained decoding - the DFA
   * will start from its initial state and generate words from scratch.
   */
  private buildPrompt(): string {
    // Empty prompt - let the constrained decoder start from initial state
    // This simplifies state management since we don't need to process prompt tokens
    return "";
  }

  /**
   * Generate a batch of words from the LLM.
   */
  private async generateBatch(): Promise<string[]> {
    if (this.state !== LLMState.READY || this.generator === null) {
      throw new LLMError(
        `Cannot generate: engine is ${this.state}`,
        this.stateError
      );
    }

    const prompt = this.buildPrompt();

    try {
      // Build generation options
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const generateOptions: Record<string, any> = {
        max_new_tokens: this.config.batchSize * 4, // ~4 tokens per word average
        temperature: this.config.temperature,
        top_p: this.config.topP,
        do_sample: true,
        return_full_text: false, // Only return generated text, not prompt
      };

      // Add constrained decoding if available
      if (this.constrainedProcessor) {
        this.constrainedProcessor.reset();
        generateOptions["logits_processor"] =
          this.constrainedProcessor.getProcessor();
      }

      // Generate text using the pipeline
      console.log(`[LLM Funbox] Generating with prompt: "${prompt}"`);
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
      const outputs = await this.generator(prompt, generateOptions);

      // Extract generated text
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
      const output = outputs[0];
      if (
        output === undefined ||
        output === null ||
        typeof output !== "object" ||
        !("generated_text" in output)
      ) {
        throw new LLMError("LLM returned unexpected output format");
      }

      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      const generatedText = output.generated_text as string;

      if (
        generatedText === undefined ||
        generatedText === null ||
        generatedText === ""
      ) {
        console.error(
          `[LLM Funbox] Generation returned empty text. ` +
            `This may indicate that constrained decoding masked all tokens.`
        );
        throw new LLMError("LLM returned empty response");
      }

      console.log(`[LLM Funbox] Generated text: "${generatedText}"`);

      // Parse the response into words
      // Remove punctuation, lowercase, split on whitespace
      const words = generatedText
        .toLowerCase()
        .replace(/[^a-z\s]/g, " ")
        .split(/\s+/)
        .filter((w) => w.length > 0 && w.length < 15); // Filter empty and overly long

      if (words.length === 0) {
        throw new LLMError("LLM response contained no valid words");
      }

      console.log(`[LLM Funbox] Generated ${words.length} words`);
      return words;
    } catch (error) {
      if (error instanceof LLMError) {
        throw error;
      }
      throw new LLMError("Failed to generate words from LLM", error);
    }
  }

  /**
   * Refill the word buffer if it's running low.
   * This is called asynchronously to avoid blocking the typing test.
   */
  private async refillBuffer(): Promise<void> {
    if (this.isGenerating) {
      return; // Already generating
    }

    if (this.wordBuffer.length >= this.config.bufferMinWords) {
      return; // Buffer is full enough
    }

    this.isGenerating = true;

    try {
      const newWords = await this.generateBatch();
      this.wordBuffer.push(...newWords);

      console.log(
        `[LLM Funbox] Buffer refilled. Size: ${this.wordBuffer.length}`
      );
    } finally {
      this.isGenerating = false;
    }
  }

  /**
   * Get the next word for the typing test.
   * This must be synchronous to match Wordset interface.
   *
   * IMPORTANT: Does NOT fallback to random words. If buffer is empty and
   * LLM is not ready, this will throw an error.
   */
  public override randomWord(): string {
    // Check for error state first
    if (this.state === LLMState.ERROR) {
      throw new LLMError("LLM engine is in error state", this.stateError);
    }

    // Try to get a word from the buffer
    if (this.wordBuffer.length > 0) {
      const word = this.wordBuffer.shift();
      if (word === undefined) {
        throw new LLMError(
          "Unexpected: buffer reported length > 0 but shift returned undefined"
        );
      }

      // Trigger async refill if buffer is getting low
      if (this.wordBuffer.length < this.config.bufferMinWords) {
        void this.refillBuffer().catch((error: unknown) => {
          console.error("[LLM Funbox] Background refill failed:", error);
          // Don't throw here - we'll throw on the next randomWord() call if buffer empties
        });
      }

      return word;
    }

    // Buffer is empty - this is a failure condition
    if (this.state === LLMState.LOADING) {
      throw new LLMError(
        "LLM is still loading. Please wait for initialization to complete."
      );
    }

    if (this.state === LLMState.UNINITIALIZED) {
      throw new LLMError("LLM has not been initialized.");
    }

    // If we get here, the buffer is empty and we can't generate more
    throw new LLMError(
      "Word buffer is empty and cannot be refilled. " +
        "This may indicate a generation failure."
    );
  }

  /**
   * Get the current state of the LLM engine
   */
  public getState(): LLMState {
    return this.state;
  }

  /**
   * Get the error if in error state
   */
  public getError(): Error | null {
    return this.stateError;
  }

  /**
   * Get the current buffer size (for debugging/UI)
   */
  public getBufferSize(): number {
    return this.wordBuffer.length;
  }

  /**
   * Reset the generator state (useful for restarting tests)
   */
  public reset(): void {
    this.wordBuffer = [];
    if (this.constrainedProcessor) {
      this.constrainedProcessor.reset();
    }
    if (this.state === LLMState.READY) {
      void this.refillBuffer();
    }
  }

  /**
   * Cleanup resources when done
   */
  public async dispose(): Promise<void> {
    this.generator = null;
    this.constrainedProcessor = null;
    this.state = LLMState.UNINITIALIZED;
    this.wordBuffer = [];
  }
}
