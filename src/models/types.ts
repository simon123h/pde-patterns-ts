export interface ParamSpec {
  key: string
  label: string
  min: number
  max: number
  step: number
  default: number
  /** The parameter only affects the initial condition, so changing it restarts the simulation. */
  resetOnChange?: boolean
}

export interface RangeSpec {
  default: number
  min: number
  max: number
  step: number
}

/** What a view shows of a (possibly complex) field. */
export type ViewKind = 're' | 'im' | 'abs' | 'arg'

export interface ViewSpec {
  label: string
  field: number
  kind: ViewKind
}

export interface InitContext {
  nx: number
  ny: number
  /** Side length of the (square) periodic domain. */
  lx: number
  params: Record<string, number>
}

/**
 * A pseudospectral PDE  ∂t û_f = L_f(k) û_f + M_f(k) · FFT[ N_f(u, ∇u) ]  for `fields`
 * fields on a periodic square domain. Each field is complex-valued in general (real models
 * simply keep the imaginary part at zero); all fields are advanced by ETDRK4.
 */
export interface SpectralModel {
  id: string
  name: string
  /** One-line description shown in the UI. */
  description: string
  fields: number
  params: ParamSpec[]
  domain: RangeSpec
  dt: RangeSpec
  /** Sensible number of solver steps per animation frame. */
  stepsPerFrame: number
  /** Real part of the linear symbol L(k²) of one field, with k² = kx² + ky². */
  linearSymbol(k2: number, field: number, params: Record<string, number>): number
  /** Imaginary part of the linear symbol (complex fields only). */
  linearSymbolImag?(k2: number, field: number, params: Record<string, number>): number
  /** Real Fourier-space multiplier M(k²) applied to the transformed nonlinearity (e.g. −k² for ∇²N). */
  nonlinearSymbol?(k2: number, field: number, params: Record<string, number>): number
  /** Whether the nonlinearity needs the spatial gradients `ux`, `uy` (otherwise they are zero). */
  usesGradient?: boolean
  /**
   * WGSL body of
   *   fn nonlinear(u: array<vec2<f32>, FIELDS>, ux: array<vec2<f32>, FIELDS>,
   *                uy: array<vec2<f32>, FIELDS>, f: u32) -> vec2<f32>
   * returning N_f pointwise as (re, im). Parameters are available as `params.<key>` (f32).
   */
  nonlinearWgsl: string
  views: ViewSpec[]
  /** Initial state: interleaved (re, im) per cell, fields back to back (length 2·fields·nx·ny). */
  initial(ctx: InitContext): Float32Array
}
