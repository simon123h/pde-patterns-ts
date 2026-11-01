import { describe, expect, it } from 'vitest'
import { phiFunctions } from '../src/math/etd'

describe('phi functions', () => {
  it('matches the z → 0 limits 1, 1/2, 1/6', () => {
    const p = phiFunctions(0)
    expect(p.phi1).toBeCloseTo(1, 12)
    expect(p.phi2).toBeCloseTo(1 / 2, 12)
    expect(p.phi3).toBeCloseTo(1 / 6, 12)
  })

  it('matches closed forms away from zero', () => {
    for (const z of [-50, -3, -0.5, 0.7]) {
      const p = phiFunctions(z)
      const e = Math.exp(z)
      expect(p.phi1).toBeCloseTo((e - 1) / z, 10)
      expect(p.phi2).toBeCloseTo((e - 1 - z) / (z * z), 10)
      expect(p.phi3).toBeCloseTo((e - 1 - z - (z * z) / 2) / z ** 3, 9)
    }
  })

  it('stays accurate for tiny |z| where the naive formula cancels', () => {
    const z = -1e-5
    const p = phiFunctions(z)
    expect(p.phi2).toBeCloseTo(0.5 + z / 6, 10)
    expect(p.phi3).toBeCloseTo(1 / 6 + z / 24, 10)
  })
})
