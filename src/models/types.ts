export interface ParamSpec {
  key: string
  label: string
  min: number
  max: number
  step: number
  default: number
}

/**
 * A pseudospectral PDE  ∂t u = L(k) û + N(u),  with a diagonal linear symbol.
 * The nonlinearity is a WGSL snippet evaluated pointwise in physical space.
 */
export interface SpectralModel {
  id: string
  name: string
  /** Number of real fields (complex models like CGLE use 2: re, im). */
  fields: number
  params: ParamSpec[]
  /** Linear symbol L(|k|²) per field, with k² = kx² + ky². Real part ≤ 0 for stable modes. */
  linearSymbol(k2: number, field: number, params: Record<string, number>): number
  /** WGSL body computing N(u) pointwise; see the solver for the bound variable names. */
  nonlinearWgsl: string
}
