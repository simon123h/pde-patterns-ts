import type { ViewKind } from '../models/types'

export interface Range {
  lo: number
  hi: number
}

/**
 * The limits the colors are actually mapped to, the same rule as in the render shader: phases always
 * span ±π, and diverging colormaps are centered on zero (range ±max(|lo|, |hi|)).
 */
export function displayRange(range: Range, kind: ViewKind, diverging: boolean): Range {
  if (kind === 'arg') return { lo: -Math.PI, hi: Math.PI }
  if (diverging) {
    const m = Math.max(Math.abs(range.lo), Math.abs(range.hi))
    return { lo: -m, hi: m }
  }
  return { ...range }
}

/** A value as short axis-label text. */
export function formatTick(value: number): string {
  if (value === 0) return '0'
  const abs = Math.abs(value)
  if (abs >= 1e4 || abs < 1e-3) return value.toExponential(1)
  const text = value.toPrecision(3)
  return text.includes('.') ? text.replace(/\.?0+$/, '') : text
}
