import { noisyState } from './initial'
import type { SpectralModel } from './types'

/** ∂t u = −∇²u − ν∇⁴u − ½ |∇u|² (2D Kuramoto–Sivashinsky) */
export const kuramotoSivashinsky: SpectralModel = {
  id: 'kuramoto-sivashinsky',
  name: 'Kuramoto–Sivashinsky',
  description: '∂t u = −∇²u − ν∇⁴u − ½|∇u|² — spatiotemporal chaos with a characteristic cell size.',
  fields: 1,
  params: [{ key: 'nu', label: 'ν (4th-order damping)', min: 0.3, max: 3, step: 0.01, default: 1 }],
  domain: { default: 100, min: 20, max: 400, step: 1 },
  dt: { default: 0.1, min: 0.01, max: 1, step: 0.01 },
  stepsPerFrame: 3,
  // ∇² → −k², ∇⁴ → k⁴
  linearSymbol: (k2, _f, p) => k2 - p.nu! * k2 * k2,
  usesGradient: true,
  nonlinearWgsl: 'return vec2<f32>(-0.5 * (ux[0].x * ux[0].x + uy[0].x * uy[0].x), 0.0);',
  views: [{ label: 'u', field: 0, kind: 're' }],
  initial: (ctx) => noisyState(ctx, [0], 0.1),
}
