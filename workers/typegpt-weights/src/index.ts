type Env = {
  TYPEGPT_WEIGHTS: R2BucketLike;
  ALLOWED_ORIGINS?: string;
};

type R2BucketLike = {
  get(key: string): Promise<R2ObjectBodyLike | null>;
};

type R2ObjectBodyLike = {
  body: ReadableStream;
  httpEtag: string;
  size: number;
};

const CACHE_CONTROL = "public, max-age=31536000, immutable";
const ALLOWED_PREFIX = "weights/gpt2/";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(request, env),
      });
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method Not Allowed", {
        status: 405,
        headers: corsHeaders(request, env),
      });
    }

    const url = new URL(request.url);
    const key = url.pathname.replace(/^\/+/, "");

    if (!key.startsWith(ALLOWED_PREFIX) || key.includes("..")) {
      return new Response("Not Found", {
        status: 404,
        headers: corsHeaders(request, env),
      });
    }

    const object = await env.TYPEGPT_WEIGHTS.get(key);
    if (object === null) {
      return new Response("Not Found", {
        status: 404,
        headers: corsHeaders(request, env),
      });
    }

    const headers = new Headers(corsHeaders(request, env));
    headers.set("Cache-Control", CACHE_CONTROL);
    headers.set("Content-Type", contentTypeForKey(key));
    headers.set("Content-Length", object.size.toString());
    headers.set("ETag", object.httpEtag);
    headers.set("Accept-Ranges", "bytes");

    if (request.method === "HEAD") {
      return new Response(null, { headers });
    }

    return new Response(object.body, { headers });
  },
};

function corsHeaders(request: Request, env: Env): HeadersInit {
  return {
    "Access-Control-Allow-Origin": allowedOrigin(request, env),
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Range",
    "Access-Control-Expose-Headers": "Content-Length, ETag",
    "Access-Control-Max-Age": "86400",
  };
}

function allowedOrigin(request: Request, env: Env): string {
  const requestOrigin = request.headers.get("Origin");
  const origins = env.ALLOWED_ORIGINS?.split(",").map((origin) =>
    origin.trim(),
  );

  if (origins === undefined || origins.length === 0) {
    return "*";
  }

  if (requestOrigin !== null && origins.includes(requestOrigin)) {
    return requestOrigin;
  }

  if (
    requestOrigin !== null &&
    isAllowedLocalhostOrigin(requestOrigin, origins)
  ) {
    return requestOrigin;
  }

  return origins[0] ?? "*";
}

function isAllowedLocalhostOrigin(
  requestOrigin: string,
  origins: string[],
): boolean {
  if (
    !origins.includes("http://localhost:*") &&
    !origins.includes("http://127.0.0.1:*")
  ) {
    return false;
  }

  try {
    const parsedOrigin = new URL(requestOrigin);

    return (
      parsedOrigin.protocol === "http:" &&
      ((parsedOrigin.hostname === "localhost" &&
        origins.includes("http://localhost:*")) ||
        (parsedOrigin.hostname === "127.0.0.1" &&
          origins.includes("http://127.0.0.1:*")))
    );
  } catch {
    return false;
  }
}

function contentTypeForKey(key: string): string {
  if (key.endsWith(".json")) {
    return "application/json; charset=utf-8";
  }

  return "application/octet-stream";
}
