import assert from 'node:assert/strict'
import test from 'node:test'
import { parseCompactProtectionFormat, incrementalExtractSuggestions } from '../src/templates/protection.js'

test('escaped commas preserve full rewritten text and removed risks', () => {
  const [value] = parseCompactProtectionFormat(String.raw`balanced,Generalized,I live in Boston\, with family.,Removed name\, address,name|address`)
  assert.equal(value.modified_text, 'I live in Boston, with family.')
  assert.equal(value.changes_summary, 'Removed name, address')
  assert.deepEqual(value.removed_risks, ['name', 'address'])
})

test('malformed complete rows cannot silently truncate rewritten text', () => {
  assert.equal(parseCompactProtectionFormat('balanced,Generalized,I live in Boston, with family.,Removed name,name'), null)
  assert.equal(parseCompactProtectionFormat('balanced,Generalized,Text,Summary'), null)
})

test('streaming still exposes partial rows until all fields arrive', () => {
  const { yielded } = incrementalExtractSuggestions('balanced,Generalized,Text')
  assert.equal(yielded[0].modified_text, 'Text')
  assert.equal(yielded[0]._isComplete, false)
})
