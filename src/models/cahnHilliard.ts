import { noisyState } from './initial'
import type { SpectralModel } from './types'

/**
 * ∂t u = ∇²(u³ − u) − γ ∇⁴u, conserving ∫u.
 *
 * Stabilized splitting: with β = 2, ∇²(u³ − u) = β∇²u + ∇²(u³ − (1+β)u), so the stiff
 * −βk² part is exact in L and the pointwise term is u³ − 3u, transformed and multiplied by −k².
 */
export const cahnHilliard: SpectralModel = {
  id: 'cahn-hilliard',
  name: 'Cahn–Hilliard',
  description: '∂t u = ∇²(u³ − u) − γ∇⁴u — spinodal decomposition and coarsening; mass is conserved.',
  fields: 1,
  params: [
    { key: 'gamma', label: 'γ (interface width²)', min: 0.1, max: 4, step: 0.05, default: 1 },
    { key: 'mean', label: 'mean concentration', min: -0.8, max: 0.8, step: 0.01, default: 0, resetOnChange: true },
  ],
  domain: { default: 128, min: 20, max: 400, step: 1 },
  dt: { default: 0.2, min: 0.01, max: 5, step: 0.01 },
  stepsPerFrame: 2,
  linearSymbol: (k2, _f, p) => -2 * k2 - p.gamma! * k2 * k2,
  nonlinearSymbol: (k2) => -k2,
  nonlinearWgsl: 'let v = u[0].x; return vec2<f32>(v * v * v - 3.0 * v, 0.0);',
  views: [{ label: 'u', field: 0, kind: 're' }],
  initial: (ctx) => noisyState(ctx, [ctx.params.mean ?? 0], 0.1),
}
