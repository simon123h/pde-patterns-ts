import { noisyState } from './initial'
import { defineModel } from './defineModel'

/**
 * Brusselator:  u_t = Du∇²u + A − (B + 1)u + u²v,  v_t = Dv∇²v + B u − u²v   (Du = 1).
 * Homogeneous state (A, B/A); Turing-unstable for B > (1 + A·√(Du/Dv))².
 */
export const brusselator = defineModel(
  {
    id: 'brusselator',
    name: 'Brusselator',
    description: 'Reaction–diffusion with Turing and Hopf instabilities — spots and stripes of wavelength ≈ 2π·(Du·Dv)^¼/√A (shows u).',
    fields: 2,
    params: [
      { key: 'A', label: 'A', min: 0.5, max: 4, step: 0.01, default: 2 },
      { key: 'B', label: 'B', min: 1, max: 9, step: 0.01, default: 3.6 },
      { key: 'Dv', label: 'Dv / Du', min: 2, max: 30, step: 0.1, default: 8 },
    ],
    domain: { default: 100, min: 20, max: 300, step: 1 },
    dt: { default: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    stepsPerFrame: 4,
    paint: { amounts: [1, 0], sigma: 0.03 },
    presets: [
      { name: 'Fine spots', description: 'Hexagonal Turing spots, well above threshold.', params: { A: 2, B: 4.6, Dv: 8 } },
      { name: 'Wide spots', description: 'Large, widely spaced spots.', params: { A: 1, B: 2.6, Dv: 8 } },
    ],
    views: [
      { label: 'u', field: 0, kind: 're' },
      { label: 'v', field: 1, kind: 're' },
    ],
  },
  {
    linearSymbol: (k2, f, p) => (f === 0 ? -k2 - (p.B! + 1) : -p.Dv! * k2),
    nonlinearWgsl: `
      let uu = u[0].x;
      let vv = u[1].x;
      let r = uu * uu * vv;
      if (f == 0u) {
        return vec2<f32>(params.A + r, 0.0);
      }
      return vec2<f32>(params.B * uu - r, 0.0);
    `,
    initial: (ctx) => noisyState(ctx, [ctx.params.A ?? 2, (ctx.params.B ?? 3.6) / (ctx.params.A ?? 2)], 0.05),
  },
)
