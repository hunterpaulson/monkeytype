/**
 * Local browser benchmark for the LLM funbox generation path.
 *
 * Examples:
 * - pnpm benchmark:llm
 * - pnpm benchmark:llm -- --headless --words 200 --window 5
 * - pnpm benchmark:llm -- --url http://127.0.0.1:3000
 */

import path from "node:path";
import process from "node:process";
import { spawn, type ChildProcess } from "node:child_process";
import { chromium } from "playwright";
import type {
  BrowserLlmBenchmarkOptions,
  BrowserLlmBenchmarkResult,
} from "../src/ts/test/llm/browser-benchmark";

type CliOptions = {
  baseUrl: string | null;
  port: number | null;
  languageFile: string | undefined;
  wordsToConsume: number | undefined;
  contextWindowSize: number | undefined;
  minP: number | undefined;
  frequencyPenalty: number | undefined;
  matrix: boolean;
  clearRuntimeCache: boolean;
  headless: boolean;
};

type DevServerHandle = {
  process: ChildProcess;
  output: string[];
  baseUrl: string;
};

const FRONTEND_ROOT = path.resolve(import.meta.dirname, "..");
const BROWSER_BENCHMARK_MODULE_PATH = "/ts/test/llm/browser-benchmark.ts";
const DEFAULT_PORT = 4173;
const SERVER_START_TIMEOUT_MS = 60_000;
const SERVER_STOP_TIMEOUT_MS = 5_000;
const SERVER_OUTPUT_LIMIT = 200;

async function main(): Promise<void> {
  const cliOptions = parseCliOptions(process.argv.slice(2));
  const benchmarkOptions: BrowserLlmBenchmarkOptions = {
    languageFile: cliOptions.languageFile,
    wordsToConsume: cliOptions.wordsToConsume,
    contextWindowSize: cliOptions.contextWindowSize,
    minP: cliOptions.minP,
    frequencyPenalty: cliOptions.frequencyPenalty,
    clearRuntimeCache: cliOptions.clearRuntimeCache,
  };

  let serverHandle: DevServerHandle | null = null;
  let browserClosed = false;
  const browser = await chromium.launch({
    headless: cliOptions.headless,
    args: ["--enable-unsafe-webgpu"],
  });

  try {
    if (cliOptions.baseUrl === null) {
      serverHandle = await startDevServer(cliOptions.port);
    }

    const baseUrl = cliOptions.baseUrl ?? serverHandle?.baseUrl;

    if (baseUrl === undefined) {
      throw new Error("Unable to determine benchmark base URL");
    }

    const page = await browser.newPage();
    page.on("console", (msg) => {
      if (msg.type() === "warning" || msg.type() === "error") {
        console.error(`[page ${msg.type()}] ${msg.text()}`);
      }
    });
    const benchmarkPageUrl = new URL("/404.html", normalizeBaseUrl(baseUrl));

    await page.goto(benchmarkPageUrl.toString(), {
      waitUntil: "domcontentloaded",
    });

    const benchmarkResults = await page.evaluate(
      async ({ modulePath, options }) => {
        const benchmarkModule = (await import(modulePath)) as {
          runLlmBrowserBenchmark(
            benchmarkOptions: BrowserLlmBenchmarkOptions,
          ): Promise<BrowserLlmBenchmarkResult>;
        };

        const benchmarkOptions = options.matrix
          ? buildBenchmarkMatrix(options.benchmarkOptions)
          : [options.benchmarkOptions];
        const results: BrowserLlmBenchmarkResult[] = [];

        for (const benchmarkOption of benchmarkOptions) {
          results.push(
            await benchmarkModule.runLlmBrowserBenchmark(benchmarkOption),
          );
        }

        return results;

        function buildBenchmarkMatrix(
          baseOptions: BrowserLlmBenchmarkOptions,
        ): BrowserLlmBenchmarkOptions[] {
          const contextWindowSizes = [5, 16, 32];
          const minPs = [0.05, 0.1];
          const frequencyPenalties = [1, 2];

          return contextWindowSizes.flatMap((contextWindowSize) =>
            minPs.flatMap((minP) =>
              frequencyPenalties.map((frequencyPenalty) => ({
                ...baseOptions,
                contextWindowSize,
                minP,
                frequencyPenalty,
                clearRuntimeCache: false,
              })),
            ),
          );
        }
      },
      {
        modulePath: BROWSER_BENCHMARK_MODULE_PATH,
        options: {
          benchmarkOptions,
          matrix: cliOptions.matrix,
        },
      },
    );
    const userAgent = await page.evaluate(() => navigator.userAgent);
    const summary = benchmarkResults.map(summarizeBenchmarkResult);

    console.log(
      JSON.stringify(
        {
          browser: {
            name: "chromium",
            version: browser.version(),
            headless: cliOptions.headless,
            userAgent,
          },
          pageUrl: benchmarkPageUrl.toString(),
          options: benchmarkOptions,
          matrix: cliOptions.matrix,
          summary,
          results: benchmarkResults,
        },
        null,
        2,
      ),
    );

    await browser.close();
    browserClosed = true;
  } finally {
    if (!browserClosed) {
      await browser.close();
    }

    await stopDevServer(serverHandle);
  }
}

function parseCliOptions(args: string[]): CliOptions {
  const options: CliOptions = {
    baseUrl: null,
    port: null,
    languageFile: undefined,
    wordsToConsume: undefined,
    contextWindowSize: undefined,
    minP: undefined,
    frequencyPenalty: undefined,
    matrix: false,
    clearRuntimeCache: false,
    headless: false,
  };

  for (let index = 0; index < args.length; index++) {
    const arg = args[index];

    if (arg === "--") {
      continue;
    }

    if (arg === "--help" || arg === "-h") {
      printUsageAndExit(0);
    }

    switch (arg) {
      case "--url":
        options.baseUrl = requireStringArg(args, ++index, arg);
        break;
      case "--port":
        options.port = requireNumberArg(args, ++index, arg);
        break;
      case "--language":
        options.languageFile = requireStringArg(args, ++index, arg);
        break;
      case "--words":
        options.wordsToConsume = requireNumberArg(args, ++index, arg);
        break;
      case "--window":
        options.contextWindowSize = requireNumberArg(args, ++index, arg);
        break;
      case "--min-p":
        options.minP = requireNumberArg(args, ++index, arg);
        break;
      case "--frequency-penalty":
        options.frequencyPenalty = requireNumberArg(args, ++index, arg);
        break;
      case "--matrix":
        options.matrix = true;
        break;
      case "--clear-cache":
        options.clearRuntimeCache = true;
        break;
      case "--headless":
        options.headless = true;
        break;
      default:
        throw new Error(`Unknown benchmark option: ${arg}`);
    }
  }

  return options;
}

function summarizeBenchmarkResult(result: BrowserLlmBenchmarkResult): object {
  return {
    contextWindowSize: result.contextWindowSize,
    minP: result.samplingParams.minP,
    frequencyPenalty: result.samplingParams.frequencyPenalty,
    wordsToConsume: result.wordsToConsume,
    wordsPerSecond: round(result.wordsPerSecond, 2),
    generatedTokensPerSecond: round(
      result.qualityMetrics.generatedTokensPerSecond,
      2,
    ),
    generatedWordsPerMinute: round(
      result.qualityMetrics.generatedWordsPerMinute,
      0,
    ),
    averageTokensPerCompletedWord: round(
      result.qualityMetrics.averageTokensPerCompletedWord,
      2,
    ),
    immediateRepeatRate: round(result.qualityMetrics.immediateRepeatRate, 3),
    repeatWithinLast10Rate: round(
      result.qualityMetrics.repeatWithinLast10Rate,
      3,
    ),
    uniqueWordRate: round(result.qualityMetrics.uniqueWordRate, 3),
    averageWordLength: round(result.qualityMetrics.averageWordLength, 2),
    topWords: result.qualityMetrics.topWords.slice(0, 5),
    sampleText: result.sampleText,
  };
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function requireStringArg(args: string[], index: number, flag: string): string {
  const value = args[index];

  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Missing value for ${flag}`);
  }

  return value;
}

function requireNumberArg(args: string[], index: number, flag: string): number {
  const value = requireStringArg(args, index, flag);
  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid numeric value for ${flag}: ${value}`);
  }

  return parsed;
}

async function startDevServer(port: number | null): Promise<DevServerHandle> {
  const requestedPort = port ?? DEFAULT_PORT;
  const serverProcess = spawn(
    resolvePnpmCommand(),
    [
      "exec",
      "vite",
      "dev",
      "--host",
      "127.0.0.1",
      "--port",
      String(requestedPort),
      ...(port === null ? [] : ["--strictPort"]),
    ],
    {
      cwd: FRONTEND_ROOT,
      env: {
        ...process.env,
        SERVER_OPEN: "false",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  const output: string[] = [];

  if (serverProcess.stdout === null || serverProcess.stderr === null) {
    throw new Error("Vite dev server output streams are unavailable");
  }

  collectProcessOutput(serverProcess.stdout, output);
  collectProcessOutput(serverProcess.stderr, output);

  const resolvedBaseUrl = await waitForServerReady(
    port === null ? null : `http://127.0.0.1:${requestedPort}`,
    serverProcess,
    output,
  );

  return { process: serverProcess, output, baseUrl: resolvedBaseUrl };
}

function collectProcessOutput(
  stream: NodeJS.ReadableStream,
  output: string[],
): void {
  stream.on("data", (chunk: Buffer | string) => {
    output.push(String(chunk));

    if (output.length > SERVER_OUTPUT_LIMIT) {
      output.splice(0, output.length - SERVER_OUTPUT_LIMIT);
    }
  });
}

async function waitForServerReady(
  preferredBaseUrl: string | null,
  serverProcess: ChildProcess,
  output: string[],
): Promise<string> {
  const deadline = Date.now() + SERVER_START_TIMEOUT_MS;

  while (Date.now() < deadline) {
    if (serverProcess.exitCode !== null) {
      throw new Error(
        `Vite dev server exited before the benchmark started.\n${output.join("")}`,
      );
    }

    const detectedBaseUrl = preferredBaseUrl ?? detectViteBaseUrl(output);

    if (detectedBaseUrl === null) {
      await delay(250);
      continue;
    }

    const benchmarkModuleUrl = new URL(
      BROWSER_BENCHMARK_MODULE_PATH,
      normalizeBaseUrl(detectedBaseUrl),
    ).toString();

    try {
      const response = await fetch(benchmarkModuleUrl);

      if (
        response.ok &&
        (response.headers.get("content-type")?.includes("javascript") ?? false)
      ) {
        return normalizeBaseUrl(detectedBaseUrl);
      }
    } catch {
      // keep polling until the dev server is ready
    }

    await delay(250);
  }

  throw new Error(
    `Timed out waiting for the benchmark dev server.\n${output.join("")}`,
  );
}

function detectViteBaseUrl(output: string[]): string | null {
  const combinedOutput = output.join("");
  const match = /https?:\/\/(?:127\.0\.0\.1|localhost):\d+\//.exec(
    combinedOutput,
  );

  return match?.[0] ?? null;
}

async function stopDevServer(
  serverHandle: DevServerHandle | null,
): Promise<void> {
  if (serverHandle === null) {
    return;
  }

  const serverProcess = serverHandle.process;

  if (serverProcess.exitCode !== null) {
    return;
  }

  serverProcess.kill("SIGTERM");

  const exited = await Promise.race([
    new Promise<boolean>((resolve) => {
      serverProcess.once("exit", () => resolve(true));
    }),
    delay(SERVER_STOP_TIMEOUT_MS).then(() => false),
  ]);

  if (!exited && serverProcess.exitCode === null) {
    serverProcess.kill("SIGKILL");
  }
}

function resolvePnpmCommand(): string {
  return process.platform === "win32" ? "pnpm.cmd" : "pnpm";
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
}

async function delay(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function printUsageAndExit(exitCode: number): never {
  console.log(`Usage: pnpm benchmark:llm -- [options]

Options:
  --url <url>         Use an existing frontend server instead of starting Vite
  --port <port>       Port for the temporary Vite dev server (default: ${DEFAULT_PORT})
  --language <file>   Language file to benchmark (default: english_5k.json)
  --words <count>     Number of generated words to consume (default: 100)
  --window <count>    Context window size override (default: 5)
  --min-p <number>    Min-p sampling value (default: 0.1)
  --frequency-penalty <number>
                      Word frequency penalty (default: 2)
  --matrix            Run the default tuning matrix
  --clear-cache       Clear the shared WebGPT runtime before benchmarking
  --headless          Run Chromium without opening a visible window
  -h, --help          Show this help message
`);
  process.exit(exitCode);
}

void main().catch((error: unknown) => {
  console.error(
    error instanceof Error ? (error.stack ?? error.message) : String(error),
  );
  process.exit(1);
});
