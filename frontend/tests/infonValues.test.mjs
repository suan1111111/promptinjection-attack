import assert from 'node:assert/strict'
import test from 'node:test'
import { parseCompactInfonsFormat } from '../src/templates/infons.js'

test('real model address with an unescaped comma keeps its value and confidence', () => {
  const [infon] = parseCompactInfonsFormat('desc:r1_3,DESC,邮寄地址,18 Willow Lane, Boston,string,0.95').infons
  assert.equal(infon.attribute, '18 Willow Lane, Boston')
  assert.equal(infon.data_type, 'string')
  assert.equal(infon.confidence, 0.95)
})

test('already escaped values retain the original parsing behavior', () => {
  const [infon] = parseCompactInfonsFormat(String.raw`desc:r1_3,DESC,邮寄地址,18 Willow Lane\, Boston,string,0.95`).infons
  assert.equal(infon.attribute, '18 Willow Lane, Boston')
  assert.equal(infon.confidence, 0.95)
})

test('ordinary numeric facts still use their declared datatype', () => {
  const [infon] = parseCompactInfonsFormat('desc:r1_2,DESC,年龄,28,number,0.95').infons
  assert.equal(infon.attribute, '28')
  assert.equal(infon.data_type, 'number')
  assert.equal(infon.confidence, 0.95)
})
