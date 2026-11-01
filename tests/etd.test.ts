import { describe, expect, it } from 'vitest'
import { krogstadStep, phiContour, phiFunctions } from '../src/math/etd'

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

describe('phi functions (fast hybrid)', () => {
  it('agrees with the contour-integral reference', () => {
    for (const z of [-1e4, -200, -12, -1.0001, -0.999, -0.2, -1e-9, 0, 1e-9, 0.3, 0.999, 1.5, 8]) {
      const a = phiFunctions(z)
      const b = phiContour(z)
      for (const key of ['phi1', 'phi2', 'phi3'] as const) {
        expect(a[key]).toBeCloseTo(b[key], 10)
      }
    }
  })
})

describe('krogstad ETDRK4', () => {
  // u' = L u + N(u), reference from many tiny classical RK4 steps
  const reference = (u0: number, L: number, N: (u: number) => number, T: number) => {
    const f = (u: number) => L * u + N(u)
    const steps = 200000
    const h = T / steps
    let u = u0
    for (let i = 0; i < steps; i++) {
      const k1 = f(u)
      const k2 = f(u + (h / 2) * k1)
      const k3 = f(u + (h / 2) * k2)
      const k4 = f(u + h * k3)
      u += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    }
    return u
  }
  const run = (u0: number, L: number, N: (u: number) => number, T: number, n: number) => {
    let u = u0
    for (let i = 0; i < n; i++) u = krogstadStep(u, L, T / n, N)
    return u
  }

  it('is exact for the linear part', () => {
    expect(run(1, -3, () => 0, 2, 4)).toBeCloseTo(Math.exp(-6), 12)
  })

  it('converges with order ~4', () => {
    const N = (u: number) => Math.sin(u) + 0.5 * u * u
    const exact = reference(0.7, -2, N, 1)
    const e1 = Math.abs(run(0.7, -2, N, 1, 64) - exact)
    const e2 = Math.abs(run(0.7, -2, N, 1, 128) - exact)
    expect(Math.log2(e1 / e2)).toBeGreaterThan(3.7)
  })

  it('stays accurate on a stiff mode with a large step', () => {
    const N = (u: number) => Math.cos(u)
    const exact = reference(0.1, -500, N, 1)
    expect(Math.abs(run(0.1, -500, N, 1, 10) - exact)).toBeLessThan(1e-3)
  })
})
