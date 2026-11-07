import { describe, expect, it } from 'vitest'
import { COLORMAPS, gradientCss } from '../src/render/colormaps'
import { displayRange, formatTick } from '../src/render/range'

describe('displayRange', () => {
  it('keeps the range for any colormap', () => {
    expect(displayRange({ lo: -1, hi: 3 }, 're')).toEqual({ lo: -1, hi: 3 })
    expect(displayRange({ lo: 0.5, hi: 2 }, 'abs')).toEqual({ lo: 0.5, hi: 2 })
  })
  it('always spans ±π for phases', () => {
    expect(displayRange({ lo: 0, hi: 1 }, 'arg')).toEqual({ lo: -Math.PI, hi: Math.PI })
  })
})

describe('formatTick', () => {
  it('uses short plain or exponential text', () => {
    expect(formatTick(0)).toBe('0')
    expect(formatTick(1)).toBe('1')
    expect(formatTick(0.25)).toBe('0.25')
    expect(formatTick(-2.5)).toBe('-2.5')
    expect(formatTick(100)).toBe('100')
    expect(formatTick(20)).toBe('20')
    expect(formatTick(10.5)).toBe('10.5')
    expect(formatTick(123456)).toBe('1.2e+5')
    expect(formatTick(0.00004)).toBe('4.0e-5')
  })
})

describe('gradientCss', () => {
  it.each(COLORMAPS.map((c) => c.id))('builds a gradient for %s', (id) => {
    const css = gradientCss(id, false)
    expect(css).toMatch(/^linear-gradient\(to top, rgb\(/)
    expect(css).toContain('100.0%')
  })
  it('has a cyclic variant for phases', () => {
    expect(gradientCss('viridis', true)).toMatch(/^linear-gradient\(to top, rgb\(/)
  })
})
