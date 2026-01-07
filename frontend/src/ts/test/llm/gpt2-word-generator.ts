import { Wordset } from "../wordset";
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  Tensor,
  env,
  type PreTrainedTokenizer,
  type PreTrainedModel,
} from "@huggingface/transformers";
import { PrecomputedConstrainedProcessor, type InitProgressCallback } from "./precomputed-constrained-processor";
import { BenchmarkContext } from "./benchmark";

export type Gpt2Config = {
  modelId: string;
  bufferMinWords: number;
  batchSize: number;
  temperature: number;
  topP: number;
  maxNewTokens: number;
  /** Fixed context window size (tokens). Keeps generation O(1) per token. */
  contextWindowSize: number;
  /** Enable benchmarking of initialization and generation */
  enableBenchmark: boolean;
  /** Use WebGPU if available */
  useWebGPU: boolean;
};

const DEFAULT_CONFIG: Gpt2Config = {
  modelId: "Xenova/distilgpt2",
  bufferMinWords: 8,
  batchSize: 20,
  temperature: 0.8,
  topP: 0.9,
  maxNewTokens: 120,
  contextWindowSize: 10, // Fixed window for O(1) generation
  enableBenchmark: false,
  useWebGPU: true, // Try WebGPU by default
};

export enum Gpt2State {
  UNINITIALIZED = "uninitialized",
  LOADING_MODEL = "loading_model",
  BUILDING_CONSTRAINTS = "building_constraints",
  PREFILLING = "prefilling",
  READY = "ready",
  ERROR = "error",
}

/**
 * Progress callback for UI updates during initialization
 */
export type Gpt2ProgressCallback = (progress: {
  state: Gpt2State;
  percent: number;
  detail?: string;
}) => void;

// Type for KV cache from transformers.js
type KVCache = Array<{
  key: Tensor;
  value: Tensor;
}>;

/**
 * Extract KV cache from model output.
 * transformers.js returns KV cache as present.X.key/value flat structure
 */
function extractKVCache(outputs: Record<string, Tensor>): KVCache | null {
  const layers: KVCache = [];
  let layerIdx = 0;
  
  while (true) {
    const keyTensor = outputs[`present.${layerIdx}.key`];
    const valueTensor = outputs[`present.${layerIdx}.value`];
    
    if (!keyTensor || !valueTensor) {
      break;
    }
    
    layers.push({ key: keyTensor, value: valueTensor });
    layerIdx++;
  }
  
  return layers.length > 0 ? layers : null;
}

/**
 * Convert KV cache to the format expected by model.forward()
 * Model expects: { past_key_values.0.key, past_key_values.0.value, ... }
 */
function kvCacheToModelInput(cache: KVCache): Record<string, Tensor> {
  const result: Record<string, Tensor> = {};
  
  cache.forEach((layer, idx) => {
    result[`past_key_values.${idx}.key`] = layer.key;
    result[`past_key_values.${idx}.value`] = layer.value;
  });
  
  return result;
}

export class Gpt2WordGenerator extends Wordset {
  private config: Gpt2Config;
  private state: Gpt2State = Gpt2State.UNINITIALIZED;
  private stateError: Error | null = null;
  private model: PreTrainedModel | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private constrainedProcessor: PrecomputedConstrainedProcessor | null = null;
  private initPromise: Promise<void> | null = null;
  private wordBuffer: string[] = [];
  private isGenerating = false;
  private vocabSize: number = 50257;

  // KV cache state for sliding window
  private kvCache: KVCache | null = null;
  private cachedTokenCount: number = 0;

  // Pre-fill to cover initial word list construction (e.g., 100+ words)
  private readonly initialBufferTarget = 150;

  // Benchmarking
  private benchmark: BenchmarkContext | null = null;

  // Progress callback
  public onProgress: Gpt2ProgressCallback | null = null;

  constructor(words: string[], config: Partial<Gpt2Config> = {}) {
    super(words);
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.length = Infinity;

    if (this.config.enableBenchmark) {
      this.benchmark = new BenchmarkContext();
    }

    // transformers.js browser prefs
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    env.useBrowserCache = true;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    env.allowLocalModels = false;

    this.initPromise = this.initEngine(words);
  }

  private async initEngine(words: string[]): Promise<void> {
    this.state = Gpt2State.LOADING_MODEL;
    this.reportProgress(0, "Loading model...");

    try {
      // Determine device
      let device: "webgpu" | "wasm" = "wasm";
      if (this.config.useWebGPU) {
        // Check WebGPU availability
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (typeof navigator !== "undefined" && "gpu" in navigator) {
          try {
            // @ts-expect-error - navigator.gpu typing
            const adapter = await navigator.gpu?.requestAdapter();
            if (adapter) {
              device = "webgpu";
              console.log("[Gpt2WordGenerator] Using WebGPU");
            }
          } catch {
            console.log("[Gpt2WordGenerator] WebGPU not available, using WASM");
          }
        }
      }

      // Load model and tokenizer separately for manual generation
      this.tokenizer = await AutoTokenizer.from_pretrained(this.config.modelId);
      this.model = await AutoModelForCausalLM.from_pretrained(this.config.modelId, {
        dtype: "fp32",
        device,
      });

      this.vocabSize = (this.model.config as { vocab_size?: number }).vocab_size ?? 50257;

      // Build constrained processor
      this.state = Gpt2State.BUILDING_CONSTRAINTS;
      this.reportProgress(30, "Building constraint processor...");

      const constraintProgress: InitProgressCallback = (progress) => {
        let percent = 30;
        if (progress.phase === "caching") {
          percent = 30 + (progress.percent * 0.3); // 30-60%
        } else if (progress.phase === "building") {
          percent = 60 + (progress.percent * 0.2); // 60-80%
        }
        this.reportProgress(percent, progress.detail);
      };

      this.constrainedProcessor = new PrecomputedConstrainedProcessor(
        words,
        this.tokenizer as unknown as {
          decode: (tokens: Array<number | bigint>, options?: { skip_special_tokens?: boolean }) => string;
        },
        this.vocabSize,
        {
          benchmark: this.benchmark ?? undefined,
          onProgress: constraintProgress,
        }
      );

      // Prefill word buffer
      this.state = Gpt2State.PREFILLING;
      this.reportProgress(80, "Generating initial words...");
      await this.prefillBuffer(this.initialBufferTarget);

      this.state = Gpt2State.READY;
      this.reportProgress(100, "Ready");

      // Log benchmark results if enabled
      if (this.benchmark) {
        console.log(this.benchmark.formatResults());
      }
    } catch (error) {
      this.state = Gpt2State.ERROR;
      this.stateError = error instanceof Error ? error : new Error(String(error));
      throw this.stateError;
    }
  }

  private reportProgress(percent: number, detail?: string): void {
    if (this.onProgress) {
      this.onProgress({
        state: this.state,
        percent,
        detail,
      });
    }
  }

  private async prefillBuffer(target: number): Promise<void> {
    const startTime = performance.now();
    let wordsGenerated = 0;

    this.benchmark?.startGeneration();

    while (this.wordBuffer.length < target) {
      const newWords = await this.generateBatch();
      this.wordBuffer.push(...newWords);
      wordsGenerated += newWords.length;

      // Report progress during prefill
      const progress = Math.min(99, Math.floor((this.wordBuffer.length / target) * 100));
      this.reportProgress(80 + (progress * 0.2), `Generated ${this.wordBuffer.length}/${target} words`);
    }

    this.benchmark?.endGeneration();

    const elapsed = performance.now() - startTime;
    console.log(
      `[Gpt2WordGenerator] Prefilled ${wordsGenerated} words in ${elapsed.toFixed(0)}ms ` +
      `(${(elapsed / wordsGenerated).toFixed(1)}ms/word)`
    );
  }

  public async waitForReady(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise;
    }
    if (this.state === Gpt2State.ERROR) {
      throw this.stateError ?? new Error("GPT-2 engine failed to initialize");
    }
  }

  /**
   * Generate a batch of words using manual token-by-token generation
   * with KV cache and sliding window for O(1) per-token performance.
   */
  private async generateBatch(): Promise<string[]> {
    if (this.model === null || this.tokenizer === null || this.constrainedProcessor === null) {
      throw new Error("Generator not ready");
    }

    const { contextWindowSize, temperature, maxNewTokens } = this.config;

    // Start with a space token
    const startTokens = this.tokenizer.encode(" ", { add_special_tokens: false }) as number[];
    let generatedIds: bigint[] = startTokens.map((t) => BigInt(t));

    this.constrainedProcessor.reset();
    
    // Reset KV cache for new batch
    this.kvCache = null;
    this.cachedTokenCount = 0;

    for (let step = 0; step < maxNewTokens; step++) {
      const { inputIds, positionOffset, shouldUseCache, shouldTrimCache } = 
        this.prepareInputsWithKVCache(generatedIds, contextWindowSize);

      // Trim KV cache if we've exceeded the window
      if (shouldTrimCache && this.kvCache) {
        this.kvCache = this.trimKVCache(this.kvCache, 1);
        this.cachedTokenCount--;
      }

      // Create tensors
      const inputTensor = new Tensor("int64", inputIds, [1, inputIds.length]);
      
      // Attention mask covers all tokens in context (cached + new)
      const totalContextLength = this.cachedTokenCount + inputIds.length;
      const attentionMask = new Tensor(
        "int64",
        new BigInt64Array(totalContextLength).fill(BigInt(1)),
        [1, totalContextLength]
      );
      
      // Position IDs for new tokens only
      const positionIds = new Tensor(
        "int64",
        inputIds.map((_, i) => BigInt(positionOffset + i)),
        [1, inputIds.length]
      );

      // Forward pass with KV cache
      // Build forward inputs - spread KV cache entries individually
      const forwardInputs: Record<string, unknown> = {
        input_ids: inputTensor,
        attention_mask: attentionMask,
        position_ids: positionIds,
        use_cache: true,
      };
      
      // Add KV cache if available (transformers.js expects past_key_values.X.key/value format)
      if (shouldUseCache && this.kvCache) {
        Object.assign(forwardInputs, kvCacheToModelInput(this.kvCache));
      }

      const outputs = await this.model.forward(forwardInputs);

      // Update KV cache (transformers.js returns present.X.key/value format)
      const newKVCache = extractKVCache(outputs as Record<string, Tensor>);
      if (newKVCache) {
        this.kvCache = newKVCache;
        this.cachedTokenCount += inputIds.length;
      }

      // Get logits for last token
      const logits = outputs.logits;
      const logitsData = logits.data as Float32Array;
      const lastTokenLogits = new Float32Array(logitsData.slice(-this.vocabSize));

      // Apply constraints
      const logitsTensor = { data: lastTokenLogits, dims: [1, this.vocabSize] };
      this.constrainedProcessor.process([generatedIds], logitsTensor);

      // Sample next token
      const nextTokenId = this.sampleToken(logitsTensor.data, temperature);
      if (nextTokenId === null) {
        console.warn("[Gpt2WordGenerator] No valid tokens, stopping batch");
        break;
      }

      generatedIds.push(BigInt(nextTokenId));
    }

    // Clear KV cache after batch
    this.kvCache = null;
    this.cachedTokenCount = 0;

    // Decode and extract words
    const generatedText = this.tokenizer.decode(
      generatedIds.map((id) => Number(id)),
      { skip_special_tokens: true }
    ) as string;

    const words = generatedText
      .split(/\s+/)
      .map((w) => w.trim().toLowerCase())
      .filter((w) => w.length > 0 && /^[a-z]+$/.test(w));

    return words;
  }

  /**
   * Prepare input IDs and determine KV cache usage for sliding window.
   */
  private prepareInputsWithKVCache(
    generatedIds: bigint[],
    contextWindowSize: number
  ): {
    inputIds: bigint[];
    positionOffset: number;
    shouldUseCache: boolean;
    shouldTrimCache: boolean;
  } {
    const totalTokens = generatedIds.length;

    // First call: no cache, process all initial tokens (up to window size)
    if (this.kvCache === null) {
      const inputIds = totalTokens > contextWindowSize
        ? generatedIds.slice(-contextWindowSize)
        : generatedIds;
      return {
        inputIds,
        positionOffset: 0,
        shouldUseCache: false,
        shouldTrimCache: false,
      };
    }

    // Subsequent calls: we have cache
    // Only need to process the newest token
    const inputIds = [generatedIds[generatedIds.length - 1]!];
    
    // Check if we need to trim the cache (exceeded window)
    const shouldTrimCache = this.cachedTokenCount >= contextWindowSize;
    
    // Position is based on how many tokens are in context
    // After trimming, we'll have contextWindowSize - 1 cached, plus 1 new = contextWindowSize
    const positionOffset = shouldTrimCache 
      ? contextWindowSize - 1  // Position of new token after trim
      : this.cachedTokenCount; // Position after existing cached tokens

    return {
      inputIds,
      positionOffset,
      shouldUseCache: true,
      shouldTrimCache,
    };
  }

  /**
   * Trim KV cache by removing the oldest N entries (for sliding window).
   * KV cache shape per layer: [batch, num_heads, seq_len, head_dim]
   */
  private trimKVCache(cache: KVCache, trimCount: number): KVCache {
    return cache.map((layer) => {
      const { key, value } = layer;
      
      // Get dimensions: [batch, num_heads, seq_len, head_dim]
      const seqLen = key.dims[2] as number;
      const newSeqLen = seqLen - trimCount;
      
      if (newSeqLen <= 0) {
        // Cache would be empty, return empty tensors
        const emptyDims = [...key.dims];
        emptyDims[2] = 0;
        return {
          key: new Tensor(key.type, new Float32Array(0), emptyDims),
          value: new Tensor(value.type, new Float32Array(0), emptyDims),
        };
      }

      // Slice out the oldest entries
      // Data layout: [batch][head][seq][dim] flattened
      const batch = key.dims[0] as number;
      const numHeads = key.dims[1] as number;
      const headDim = key.dims[3] as number;
      
      const keyData = key.data as Float32Array;
      const valueData = value.data as Float32Array;
      
      const newKeyData = new Float32Array(batch * numHeads * newSeqLen * headDim);
      const newValueData = new Float32Array(batch * numHeads * newSeqLen * headDim);

      // Copy data, skipping first trimCount positions
      for (let b = 0; b < batch; b++) {
        for (let h = 0; h < numHeads; h++) {
          for (let s = 0; s < newSeqLen; s++) {
            const oldSeqIdx = s + trimCount;
            for (let d = 0; d < headDim; d++) {
              const oldIdx = ((b * numHeads + h) * seqLen + oldSeqIdx) * headDim + d;
              const newIdx = ((b * numHeads + h) * newSeqLen + s) * headDim + d;
              newKeyData[newIdx] = keyData[oldIdx]!;
              newValueData[newIdx] = valueData[oldIdx]!;
            }
          }
        }
      }

      return {
        key: new Tensor(key.type, newKeyData, [batch, numHeads, newSeqLen, headDim]),
        value: new Tensor(value.type, newValueData, [batch, numHeads, newSeqLen, headDim]),
      };
    });
  }

  /**
   * Sample a token from masked logits using temperature sampling
   */
  private sampleToken(maskedLogits: Float32Array, temperature: number): number | null {
    // Find valid tokens
    const validIndices: number[] = [];
    const validLogits: number[] = [];

    for (let i = 0; i < this.vocabSize; i++) {
      const logit = maskedLogits[i];
      if (logit !== undefined && logit !== -Infinity && !isNaN(logit)) {
        validIndices.push(i);
        validLogits.push(logit / temperature);
      }
    }

    if (validIndices.length === 0) {
      return null;
    }

    // Softmax
    const maxLogit = Math.max(...validLogits);
    const expLogits = validLogits.map((l) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((e) => e / sumExp);

    // Sample
    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i]!;
      if (r <= 0) {
        return validIndices[i]!;
      }
    }

    return validIndices[validIndices.length - 1]!;
  }

  private async refillBuffer(): Promise<void> {
    if (this.isGenerating) return;
    if (this.wordBuffer.length >= this.config.bufferMinWords) return;

    this.isGenerating = true;
    try {
      const newWords = await this.generateBatch();
      this.wordBuffer.push(...newWords);
    } finally {
      this.isGenerating = false;
    }
  }

  public override randomWord(): string {
    if (this.state === Gpt2State.ERROR) {
      throw this.stateError ?? new Error("GPT-2 engine error");
    }

    if (this.wordBuffer.length === 0) {
      if (this.state !== Gpt2State.READY) {
        throw new Error(`GPT-2 engine not ready (state: ${this.state})`);
      }
      throw new Error("Word buffer is empty");
    }

    const word = this.wordBuffer.shift();
    if (word === undefined) {
      throw new Error("Buffer underflow");
    }

    if (this.wordBuffer.length < this.config.bufferMinWords) {
      void this.refillBuffer();
    }

    return word;
  }

  public getState(): Gpt2State {
    return this.state;
  }

  public getBufferSize(): number {
    return this.wordBuffer.length;
  }

  /**
   * Get benchmark results (if benchmarking was enabled)
   */
  public getBenchmarkResults(): ReturnType<BenchmarkContext["getResult"]> | null {
    return this.benchmark?.getResult() ?? null;
  }

  public reset(): void {
    // Clear KV cache
    this.kvCache = null;
    this.cachedTokenCount = 0;
    
    if (this.constrainedProcessor) {
      this.constrainedProcessor.reset();
    }
    if (
      this.state === Gpt2State.READY &&
      this.wordBuffer.length < this.config.bufferMinWords
    ) {
      void this.prefillBuffer(this.initialBufferTarget);
    }
  }

  public async dispose(): Promise<void> {
    this.model = null;
    this.tokenizer = null;
    this.constrainedProcessor = null;
    this.wordBuffer = [];
    this.kvCache = null;
    this.cachedTokenCount = 0;
    this.state = Gpt2State.UNINITIALIZED;
  }
}
