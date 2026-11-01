import { describe, expect, it } from 'vitest'
import { fft2d } from '../src/math/fft'

describe('cpu fft', () => {
  it('round-trips a 2D field', () => {
    const nx = 16
    const ny = 8
    const re = new Float64Array(nx * ny).map(() => Math.random())
    const im = new Float64Array(nx * ny)
    const orig = re.slice()
    fft2d(re, im, nx, ny)
    fft2d(re, im, nx, ny, true)
    for (let i = 0; i < re.length; i++) expect(re[i]! / (nx * ny)).toBeCloseTo(orig[i]!, 12)
  })

  it('puts a single cosine in the right Fourier modes', () => {
    const n = 32
    const re = new Float64Array(n * n)
    const im = new Float64Array(n * n)
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) re[y * n + x] = Math.cos((2 * Math.PI * 3 * x) / n)
    fft2d(re, im, n, n)
    expect(re[3]).toBeCloseTo((n * n) / 2, 8)
    expect(re[n - 3]).toBeCloseTo((n * n) / 2, 8)
    expect(re[4]).toBeCloseTo(0, 8)
  })
})
