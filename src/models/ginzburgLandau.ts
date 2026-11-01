import { noisyState } from './initial'
import type { SpectralModel } from './types'

/** ∂t A = A + (1 + iα)∇²A − (1 + iβ)|A|²A, A complex */
export const ginzburgLandau: SpectralModel = {
  id: 'cgle',
  name: 'Complex Ginzburg–Landau',
  description: '∂t A = A + (1+iα)∇²A − (1+iβ)|A|²A — spirals, defect turbulence and phase turbulence (Benjamin–Feir unstable for 1+αβ < 0).',
  fields: 1,
  params: [
    { key: 'alpha', label: 'α (dispersion)', min: -4, max: 4, step: 0.01, default: 1.5 },
    { key: 'beta', label: 'β (nonlinear frequency)', min: -4, max: 4, step: 0.01, default: -1.5 },
  ],
  domain: { default: 128, min: 20, max: 400, step: 1 },
  dt: { default: 0.1, min: 0.01, max: 1, step: 0.01 },
  stepsPerFrame: 2,
  // ∇² → −k²:  L = 1 − (1 + iα) k²
  linearSymbol: (k2) => 1 - k2,
  linearSymbolImag: (k2, _f, p) => -p.alpha! * k2,
  nonlinearWgsl: `
    let a = u[0];
    let m = dot(a, a);
    // −(1 + iβ) m A
    return -m * vec2<f32>(a.x - params.beta * a.y, a.y + params.beta * a.x);
  `,
  views: [
    { label: '|A|', field: 0, kind: 'abs' },
    { label: 'Re A', field: 0, kind: 're' },
    { label: 'arg A', field: 0, kind: 'arg' },
  ],
  initial: (ctx) => noisyState(ctx, [0], 0.1, true),
}
