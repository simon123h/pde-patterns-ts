import { KROGSTAD_FLOATS, krogstadInto } from '../math/etd'
import type { SpectralModel } from '../models/types'

export interface GridSpec {
  nx: number
  ny: number
  /** Grid spacing; the domain is [0, nx·dx) × [0, ny·dx), periodic. */
  dx: number
}

/** Number of complex coefficient planes per field: E, E2, A1, B2, C1, C2, F1, F2, F3. */
export const COEFFICIENT_PLANES = 9

/** Spectral modes per field: the Hermitian half-plane kx ≤ nx/2 for real fields, everything for complex ones. */
export function spectralModes(model: SpectralModel, grid: GridSpec): number {
  return (model.complex ? grid.nx : grid.nx / 2 + 1) * grid.ny
}

/** Signed integer wavenumber index of FFT bin i of n. */
const signedIndex = (i: number, n: number) => (i < n / 2 ? i : i - n)

/**
 * Per-mode ETDRK4 coefficients in the GPU's spectral layout (index = field·M + kx·ny + ky,
 * M = spectralModes), stored as COEFFICIENT_PLANES consecutive planes of fields·M complex numbers
 * (interleaved re, im). The coefficients that multiply the nonlinear terms (A1…F3) include the
 * model's Fourier-space multiplier and are zeroed outside the 2/3 dealiasing box.
 */
export function buildCoefficients(
  model: SpectralModel,
  params: Record<string, number>,
  grid: GridSpec,
  h: number,
): Float32Array<ArrayBuffer> {
  const { nx, ny, dx } = grid
  const kxCount = model.complex ? nx : nx / 2 + 1
  const M = kxCount * ny
  const total = model.fields * M
  const out = new Float32Array(2 * COEFFICIENT_PLANES * total)
  const kUnitX = (2 * Math.PI) / (nx * dx)
  const kUnitY = (2 * Math.PI) / (ny * dx)
  const c = new Float64Array(KROGSTAD_FLOATS)

  for (let field = 0; field < model.fields; field++) {
    for (let kx = 0; kx < kxCount; kx++) {
      const mx = signedIndex(kx, nx)
      for (let ky = 0; ky < ny; ky++) {
        const my = signedIndex(ky, ny)
        const dealiased = Math.abs(mx) * 3 > nx || Math.abs(my) * 3 > ny
        const k2 = (mx * kUnitX) ** 2 + (my * kUnitY) ** 2
        const kxv = mx * kUnitX
        const kyv = my * kUnitY
        krogstadInto(c, model.linearSymbol(k2, field, params, kxv, kyv), model.linearSymbolImag?.(k2, field, params, kxv, kyv) ?? 0, h)
        const mask = dealiased ? 0 : (model.nonlinearSymbol?.(k2, field, params, kxv, kyv) ?? 1)
        const g = field * M + kx * ny + ky
        for (let plane = 0; plane < COEFFICIENT_PLANES; plane++) {
          const scale = plane < 2 ? 1 : mask // E, E2 act on the state; the rest on N
          const o = 2 * (plane * total + g)
          out[o] = c[2 * plane]! * scale
          out[o + 1] = c[2 * plane + 1]! * scale
        }
      }
    }
  }
  return out
}
