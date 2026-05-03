# typeGPT

typeGPT is an unofficial Monkeytype fork for typing against GPT-2 text generated locally in the browser. The demo loads GPT-2 weights into WebGPU and uses a constrained decoder so the model emits words from Monkeytype word lists instead of arbitrary free-form text.

The public demo is intentionally small: no accounts, no leaderboards, no result saving, and no server-side inference. The point is to make the core idea easy to try: can you type faster than a local GPT-2 model can generate the next test?

## Why this exists

Typing tests usually sample words uniformly or from static quotes. This project explores a different shape: using an LLM as the word generator while preserving Monkeytype's typing-test interface and constraints.

The goal is not perfect prose. The goal is typing-useful naturalness: word transitions that feel more like English than random sampling, low repetition, good typing rhythm, and fast enough local inference that the model generation itself becomes part of the demo.

## What it does

- Runs GPT-2 inference in the browser with WebGPU.
- Constrains generation to Monkeytype language word sets.
- Streams generated words into the typing test as needed.
- Shows live generation throughput as `tok/s = generated wpm`.
- Defaults to 15-second tests so each restart shows GPT-2 decoding a fresh sequence.
- Keeps generation and typed text on device.

## Status

This is a standalone demo branch, not an endorsed Monkeytype release. The original upstream pull request was [monkeytypegame/monkeytype#7771](https://github.com/monkeytypegame/monkeytype/pull/7771).

The current implementation is useful as a demo and research prototype. Known rough edges:

- Model weights currently load from a GitHub-hosted weights repository.
- Generated text still has word attractors, where a few words appear more often than they should.
- Browser support depends on WebGPU; Chrome and Edge are the safest choices.
- Headless/browser benchmark performance can differ from interactive Chrome performance.

## Running locally

Use Node 24 and pnpm.

```bash
cd /path/to/monkeytype
source ~/.nvm/nvm.sh
nvm use 24.11.0
pnpm install
TYPEGPT_DEMO=true pnpm --filter @monkeytype/frontend dev
```

Then open:

```text
http://localhost:3000
```

If port 3000 is occupied, Vite will print the URL it selected.

## Production build

The Cloudflare Worker/static-assets build uses:

```bash
pnpm run build-typegpt
npx wrangler deploy
```

The Worker config is in `wrangler.toml`, and the static output is `frontend/dist`.

## Benchmarking

The repo includes a browser benchmark for LLM generation quality and speed.

Run one config:

```bash
pnpm --filter @monkeytype/frontend benchmark:llm -- --url http://127.0.0.1:3000 --words 80 --window 5 --min-p 0.05 --frequency-penalty 2 --headless
```

Run the default tuning matrix:

```bash
pnpm --filter @monkeytype/frontend benchmark:llm -- --url http://127.0.0.1:3000 --words 80 --matrix --headless
```

The benchmark reports throughput, generated words per minute, tokens per completed word, local repeat rates, uniqueness, word-length texture, and sample text for manual review.

## Attribution

typeGPT is based on [Monkeytype](https://monkeytype.com), an open-source typing test by the Monkeytype contributors. This fork preserves Monkeytype's GPLv3 license and links back to the original project.

This project is not affiliated with or endorsed by Monkeytype, OpenAI, or Anthropic.
