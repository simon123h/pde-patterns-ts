import { noisyState } from './initial'
import type { SpectralModel } from './types'

/**
 * Nikolaevskiy equation (2D form), for the potential u of the original velocity-like field:
 *   ∂t u = −∇²[ε − (1 + ∇²)²] u − ½ |∇u|²
 * Long waves (k → 0) are only weakly damped, which gives "soft-mode" (weak) turbulence just
 * above the onset ε = 0.
 */
export const nikolaevskiy: SpectralModel = {
  id: 'nikolaevskiy',
  name: 'Nikolaevskiy',
  description: '∂t u = −∇²[ε − (1+∇²)²]u − ½|∇u|² — oscillating stripes and soft-mode turbulence (λ = 2π, ε > 0 unstable).',
  fields: 1,
  params: [{ key: 'eps', label: 'ε', min: -0.3, max: 1, step: 0.01, default: 0.3 }],
  domain: { default: 128, min: 20, max: 400, step: 1 },
  dt: { default: 0.1, min: 0.01, max: 1, step: 0.01 },
  stepsPerFrame: 3,
  // ∇² → −k², so L = k² [ε − (1 − k²)²]
  linearSymbol: (k2, _f, p) => k2 * (p.eps! - (1 - k2) ** 2),
  usesGradient: true,
  nonlinearWgsl: 'return vec2<f32>(-0.5 * (ux[0].x * ux[0].x + uy[0].x * uy[0].x), 0.0);',
  paint: { amounts: [1], sigma: 0.03 },
  views: [{ label: 'u', field: 0, kind: 're' }],
  initial: (ctx) => noisyState(ctx, [0], 0.1),
}
