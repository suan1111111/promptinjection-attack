import assert from 'node:assert/strict'
import test from 'node:test'
import { getInferenceTimeoutMs, isRunTimedOut } from '../src/config/localRuntime.js'

test('local cold model load survives the page one-minute cleanup', () => {
  const timeout = getInferenceTimeoutMs('qwen3-4b-instruct-2507', 'http://127.0.0.1:11434/v1', 60000)
  const run = { status: 'running', createdAt: 1000, requestTimeoutMs: timeout }
  assert.equal(isRunTimedOut(run, 121000, 60000), false)
  assert.equal(isRunTimedOut(run, 1000 + timeout + 1, 60000), true)
})

test('remote providers and ordinary Ollama models keep their original timeout', () => {
  assert.equal(getInferenceTimeoutMs('qwen3-4b-instruct-2507', 'https://provider.example/v1', 60000), 60000)
  assert.equal(getInferenceTimeoutMs('gemma3:12b', 'http://localhost:11434/v1', 90000), 90000)
  assert.equal(isRunTimedOut({ status: 'running', createdAt: 1000 }, 62000, 60000), true)
})

test('completed runs are never aborted by age', () => {
  assert.equal(isRunTimedOut({ status: 'done', createdAt: 1000 }, 9999999, 60000), false)
})
