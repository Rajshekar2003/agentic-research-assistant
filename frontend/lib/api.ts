/** A source document returned alongside a research answer. */
export interface Source {
  title: string;
  url: string;
  snippet: string;
}

/** Full response shape returned by POST /research. */
export interface ResearchResponse {
  answer: string;
  sources: Source[];
  elapsed_ms: number;
  mode: 'baseline' | 'graph';
}

/**
 * Submit a natural-language query to the backend research endpoint.
 *
 * @param query - The natural-language research question.
 * @param mode  - Which pipeline to use: "baseline" hits POST /research,
 *                "graph" hits POST /research/graph. Defaults to "baseline".
 *
 * Applies a 30-second AbortController timeout. Throws an Error on network
 * failure, timeout, or non-2xx HTTP status. The error message is extracted
 * from the response body's `detail` field when available.
 */
export async function runResearch(
  query: string,
  mode: 'baseline' | 'graph' = 'baseline',
): Promise<ResearchResponse> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30_000);
  const url = mode === 'graph' ? `${apiUrl}/research/graph` : `${apiUrl}/research`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let message = `Request failed (status ${response.status})`;
      try {
        const body = await response.json();
        if (body.detail) message = String(body.detail);
      } catch {
        // keep default message
      }
      throw new Error(message);
    }

    return response.json() as Promise<ResearchResponse>;
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Request timed out');
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}
