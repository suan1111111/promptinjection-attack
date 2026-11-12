const localModels = new Set([
  'qwen2.5-3b-instruct', 'qwen2.5-vl-3b-instruct', 'qwen3-4b-instruct-2507',
])

export function getInferenceTimeoutMs(model, baseUrl, fallbackMs) {
  try {
    const url = new URL(baseUrl)
    if (localModels.has(model) && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) && url.port === '11434') {
      const configured = Number(import.meta.env?.VITE_LOCAL_MODEL_TIMEOUT_MS)
      return Number.isFinite(configured) && configured > 0 ? configured : 20 * 60 * 1000
    }
  } catch (_) {}
  return fallbackMs
}

export function isRunTimedOut(run, now, maxAgeMs) {
  return run.status === 'running' && run.createdAt &&
    now - run.createdAt > Math.max(maxAgeMs, run.requestTimeoutMs || 0)
}
