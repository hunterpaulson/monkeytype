import { env, pipeline, TextGenerationPipeline } from "@huggingface/transformers";
import { Gpt2ConstrainedLogitsProcessor } from "./gpt2-constrained-logits-processor";

env.useBrowserCache = false;
env.allowLocalModels = false;

const wordset = ["hello", "world", "help", "word", "type", "monkey", "test"];

async function main(): Promise<void> {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
  const gen = await pipeline("text-generation", "Xenova/distilgpt2");
  const generator = gen as TextGenerationPipeline;

  const tokenizer = (generator as unknown as { tokenizer?: unknown }).tokenizer;
  const vocabSize =
    (generator as unknown as { model?: { config?: { vocab_size?: number } } }).model?.config
      ?.vocab_size ?? 50257;

  if (!tokenizer) {
    // eslint-disable-next-line no-console
    console.error("Tokenizer unavailable");
    return;
  }

  const processor = new Gpt2ConstrainedLogitsProcessor(
    wordset,
    tokenizer as never,
    vocabSize as number,
    { debug: true },
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const outputs = await generator(" ", {
    max_new_tokens: 60,
    temperature: 0.9,
    top_p: 0.9,
    do_sample: true,
    return_full_text: false,
    logits_processor: [processor.getProcessor()],
  } as any);

  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const generated = (outputs as Array<{ generated_text?: string }>)?.[0]?.generated_text;
  // eslint-disable-next-line no-console
  console.log("Generated:", generated);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
if (typeof process !== "undefined" && (process as any).argv) {
  // eslint-disable-next-line @typescript-eslint/no-floating-promises
  main();
}

export { main as debugGpt2Constrained };

