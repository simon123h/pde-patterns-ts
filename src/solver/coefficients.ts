import { krogstadCoefficients } from '../math/etd'
import type { SpectralModel } from '../models/types'

export interface GridSpec {
  nx: number
  ny: number
  /** Grid spacing; the domain is [0, nx·dx) × [0, ny·dx), periodic. */
  dx: number
}

/** Number of coefficient planes per field: E, E2, A1, B2, C1, C2, F1, F2, F3. */
export const COEFFICIENT_PLANES = 9

/** Signed integer wavenumber index of FFT bin i of n. */
const signedIndex = (i: number, n: number) => (i < n / 2 ? i : i - n)

/**
 * Per-mode ETDRK4 coefficients in the GPU's spectral layout (index = field·M + kx·ny + ky,
 * M = nx·ny), stored as COEFFICIENT_PLANES consecutive planes of fields·M floats.
 * Modes outside the 2/3 dealiasing box have their nonlinear coefficients zeroed.
 */
export function buildCoefficients(
  model: SpectralModel,
  params: Record<string, number>,
  grid: GridSpec,
  h: number,
): Float32Array<ArrayBuffer> {
  const { nx, ny, dx } = grid
  const M = nx * ny
  const total = model.fields * M
  const out = new Float32Array(COEFFICIENT_PLANES * total)
  const kUnitX = (2 * Math.PI) / (nx * dx)
  const kUnitY = (2 * Math.PI) / (ny * dx)

  for (let field = 0; field < model.fields; field++) {
    for (let kx = 0; kx < nx; kx++) {
      const mx = signedIndex(kx, nx)
      for (let ky = 0; ky < ny; ky++) {
        const my = signedIndex(ky, ny)
        const dealiased = Math.abs(mx) * 3 > nx || Math.abs(my) * 3 > ny
        const k2 = (mx * kUnitX) ** 2 + (my * kUnitY) ** 2
        const c = krogstadCoefficients(model.linearSymbol(k2, field, params), h)
        const g = field * M + kx * ny + ky
        const mask = dealiased ? 0 : 1
        out[0 * total + g] = c.E
        out[1 * total + g] = c.E2
        out[2 * total + g] = c.A1 * mask
        out[3 * total + g] = c.B2 * mask
        out[4 * total + g] = c.C1 * mask
        out[5 * total + g] = c.C2 * mask
        out[6 * total + g] = c.F1 * mask
        out[7 * total + g] = c.F2 * mask
        out[8 * total + g] = c.F3 * mask
      }
    }
  }
  return out
}
