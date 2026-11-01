import { describe, expect, it } from 'vitest'
import { KROGSTAD_FLOATS, krogstadCoefficients, krogstadInto, krogstadStep, phiComplexInto, phiContour, phiFunctions } from '../src/math/etd'

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

describe('complex coefficients', () => {
  it('reduce to the real ones for a real symbol', () => {
    const out = new Float64Array(KROGSTAD_FLOATS)
    for (const L of [-300, -3, -0.4, 0, 0.2]) {
      krogstadInto(out, L, 0, 0.3)
      const r = krogstadCoefficients(L, 0.3)
      const real = [r.E, r.E2, r.A1, r.B2, r.C1, r.C2, r.F1, r.F2, r.F3]
      real.forEach((v, i) => {
        expect(out[2 * i]).toBeCloseTo(v, 12)
        expect(out[2 * i + 1]).toBe(0)
      })
    }
  })

  it('has matching phi values across the |z| = 1 switch', () => {
    const below = new Float64Array(6)
    const above = new Float64Array(6)
    phiComplexInto(0.6, 0.79, below) // |z| ≈ 0.992
    phiComplexInto(0.6, 0.81, above) // |z| ≈ 1.008
    for (let i = 0; i < 6; i++) expect(Math.abs(below[i]! - above[i]!)).toBeLessThan(0.05)
    // direct check against the closed form for a complex z
    const z = [-0.7, 2.1]
    const out = new Float64Array(6)
    phiComplexInto(z[0]!, z[1]!, out)
    const ex = Math.exp(z[0]!)
    const emr = ex * Math.cos(z[1]!) - 1
    const emi = ex * Math.sin(z[1]!)
    const d = z[0]! ** 2 + z[1]! ** 2
    expect(out[0]).toBeCloseTo((emr * z[0]! + emi * z[1]!) / d, 12)
    expect(out[1]).toBeCloseTo((emi * z[0]! - emr * z[1]!) / d, 12)
  })

  it('integrates a complex scalar ODE with 4th order', () => {
    // u' = L u + N(u), L = −1 + 3i, N = −(1 + 0.5i)|u|² u
    const Lr = -1
    const Li = 3
    const N = (r: number, i: number): [number, number] => {
      const m = r * r + i * i
      return [-m * (r - 0.5 * i), -m * (i + 0.5 * r)]
    }
    const f = (r: number, i: number): [number, number] => {
      const n = N(r, i)
      return [Lr * r - Li * i + n[0], Lr * i + Li * r + n[1]]
    }
    let [rr, ri] = [0.8, 0.3]
    const S = 100000
    const hh = 1 / S
    for (let s = 0; s < S; s++) {
      const k1 = f(rr, ri)
      const k2 = f(rr + (hh / 2) * k1[0], ri + (hh / 2) * k1[1])
      const k3 = f(rr + (hh / 2) * k2[0], ri + (hh / 2) * k2[1])
      const k4 = f(rr + hh * k3[0], ri + hh * k3[1])
      rr += (hh / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
      ri += (hh / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    }
    const cm = (a: number, b: number, c: number, d: number): [number, number] => [a * c - b * d, a * d + b * c]
    const run = (n: number) => {
      const h = 1 / n
      const c = new Float64Array(KROGSTAD_FLOATS)
      krogstadInto(c, Lr, Li, h)
      const k = (j: number): [number, number] => [c[2 * j]!, c[2 * j + 1]!]
      const add = (...v: [number, number][]): [number, number] => v.reduce((s, t) => [s[0] + t[0], s[1] + t[1]], [0, 0])
      let u: [number, number] = [0.8, 0.3]
      for (let s = 0; s < n; s++) {
        const Nn = N(...u)
        const a = add(cm(...k(1), ...u), cm(...k(2), ...Nn))
        const Na = N(...a)
        const b = add(cm(...k(1), ...u), cm(...k(2), ...Nn), cm(...k(3), Na[0] - Nn[0], Na[1] - Nn[1]))
        const Nb = N(...b)
        const cc = add(cm(...k(0), ...u), cm(...k(4), ...Nn), cm(...k(5), Nb[0] - Nn[0], Nb[1] - Nn[1]))
        const Nc = N(...cc)
        u = add(cm(...k(0), ...u), cm(...k(6), ...Nn), cm(...k(7), Na[0] + Nb[0], Na[1] + Nb[1]), cm(...k(8), ...Nc))
      }
      return Math.hypot(u[0] - rr, u[1] - ri)
    }
    expect(Math.log2(run(64) / run(128))).toBeGreaterThan(3.6)
  })
})
