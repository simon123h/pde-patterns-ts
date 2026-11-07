import type { ViewKind } from '../models/types'

export interface Range {
  lo: number
  hi: number
}

/**
 * The limits the colors are actually mapped to, the same rule as in the render shader: phases always
 * span ±π, and all other views map linearly between lo and hi.
 */
export function displayRange(range: Range, kind: ViewKind): Range {
  if (kind === 'arg') return { lo: -Math.PI, hi: Math.PI }
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
