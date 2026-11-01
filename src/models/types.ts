export interface ParamSpec {
  key: string
  label: string
  min: number
  max: number
  step: number
  default: number
}

/**
 * A pseudospectral PDE  ∂t u = L û + N(u)  with a diagonal linear symbol, for real fields
 * u = (u[0], …, u[fields − 1]) on a periodic domain.
 */
export interface SpectralModel {
  id: string
  name: string
  /** Number of real fields. */
  fields: number
  params: ParamSpec[]
  /** Linear symbol L(k²) of one field, with k² = kx² + ky². Real part ≤ 0 for stable modes. */
  linearSymbol(k2: number, field: number, params: Record<string, number>): number
  /**
   * WGSL body of `fn nonlinear(u: array<f32, FIELDS>, f: u32) -> f32`, returning N_f(u)
   * pointwise in physical space. Parameters are available as `params.<key>` (f32).
   */
  nonlinearWgsl: string
  /** Standard deviation of the white-noise initial condition, per field. */
  noiseAmplitude: number
}
