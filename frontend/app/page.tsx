'use client';

import React, { useState, useRef } from 'react';
import { runResearch, type ResearchResponse } from '@/lib/api';

export default function Home() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await runResearch(query.trim());
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      formRef.current?.requestSubmit();
    }
  }

  return (
    <main className="max-w-2xl mx-auto py-12 px-4">
      <h1 className="text-3xl font-bold mb-2">Agentic Research Assistant</h1>
      <p className="text-gray-600 mb-8">
        Ask anything. The system will search the web and synthesize an answer.
      </p>

      <form ref={formRef} onSubmit={handleSubmit}>
        <label htmlFor="query" className="sr-only">
          Your question
        </label>
        <textarea
          id="query"
          rows={4}
          className="w-full border rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-black"
          placeholder="What would you like to research?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="bg-black text-white rounded-lg px-6 py-2 mt-4 disabled:opacity-50 hover:bg-gray-800 transition"
        >
          {loading ? 'Researching...' : 'Research'}
        </button>
      </form>

      {loading && (
        <div aria-live="polite" className="flex items-center gap-2 mt-6">
          <div className="animate-spin border-2 border-gray-300 border-t-black w-4 h-4 rounded-full" />
          <span>Researching...</span>
        </div>
      )}

      {result && !loading && (
        <>
          <div className="border rounded-lg p-4 mt-6">
            <p className="whitespace-pre-wrap">{result.answer}</p>
            <div className="flex items-center gap-2 mt-3">
              <span className="text-sm text-gray-500">Answered in {result.elapsed_ms}ms</span>
              <span className="rounded-full px-2 py-0.5 bg-gray-100 text-gray-700 text-xs">
                {result.mode}
              </span>
            </div>
          </div>

          {result.sources.length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-700 mt-6 mb-2">Sources</p>
              {result.sources.map((source, index) => (
                <div
                  key={source.url}
                  className="border rounded-lg p-3 mb-2 hover:border-gray-400 transition"
                >
                  <div>
                    <span className="rounded px-2 py-0.5 bg-gray-100 text-gray-700 text-xs font-mono mr-2">
                      [{index + 1}]
                    </span>
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline font-medium"
                    >
                      {source.title}
                    </a>
                  </div>
                  <p className="text-xs text-gray-500 truncate mt-1">{source.url}</p>
                  <p className="text-sm text-gray-700 mt-1 line-clamp-3">{source.snippet}</p>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {error && !loading && (
        <div
          role="alert"
          className="border border-red-300 bg-red-50 rounded-lg p-4 mt-6 text-red-900"
        >
          {error}
        </div>
      )}
    </main>
  );
}
