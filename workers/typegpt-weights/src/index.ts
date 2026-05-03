type Env = {
  TYPEGPT_WEIGHTS: R2BucketLike;
  ALLOWED_ORIGIN?: string;
};

type R2BucketLike = {
  get(key: string): Promise<R2ObjectBodyLike | null>;
};

type R2ObjectBodyLike = {
  body: ReadableStream;
  httpEtag: string;
};

const CACHE_CONTROL = "public, max-age=31536000, immutable";
const ALLOWED_PREFIX = "weights/gpt2/";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(env),
      });
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method Not Allowed", {
        status: 405,
        headers: corsHeaders(env),
      });
    }

    const url = new URL(request.url);
    const key = url.pathname.replace(/^\/+/, "");

    if (!key.startsWith(ALLOWED_PREFIX) || key.includes("..")) {
      return new Response("Not Found", {
        status: 404,
        headers: corsHeaders(env),
      });
    }

    const object = await env.TYPEGPT_WEIGHTS.get(key);
    if (object === null) {
      return new Response("Not Found", {
        status: 404,
        headers: corsHeaders(env),
      });
    }

    const headers = new Headers(corsHeaders(env));
    headers.set("Cache-Control", CACHE_CONTROL);
    headers.set("Content-Type", contentTypeForKey(key));
    headers.set("ETag", object.httpEtag);
    headers.set("Accept-Ranges", "bytes");

    if (request.method === "HEAD") {
      return new Response(null, { headers });
    }

    return new Response(object.body, { headers });
  },
};

function corsHeaders(env: Env): HeadersInit {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN ?? "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Range",
    "Access-Control-Max-Age": "86400",
  };
}

function contentTypeForKey(key: string): string {
  if (key.endsWith(".json")) {
    return "application/json; charset=utf-8";
  }

  return "application/octet-stream";
}
