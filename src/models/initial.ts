import { gaussian } from '../math/random'
import type { InitContext } from './types'

/** Constant `mean[f]` plus white noise of standard deviation `amplitude` in the real part of each field. */
export function noisyState(ctx: InitContext, mean: number[], amplitude: number, complexNoise = false): Float32Array {
  const cells = ctx.nx * ctx.ny
  const out = new Float32Array(2 * mean.length * cells)
  mean.forEach((m, f) => {
    for (let i = 0; i < cells; i++) {
      out[2 * (f * cells + i)] = m + amplitude * gaussian()
      if (complexNoise) out[2 * (f * cells + i) + 1] = amplitude * gaussian()
    }
  })
  return out
}
