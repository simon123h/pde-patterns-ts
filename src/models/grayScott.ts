import { gaussian } from '../math/random'
import type { SpectralModel } from './types'

const DU = 2e-5
const DV = 1e-5

/**
 * Gray–Scott in Pearson's units:  u_t = Du∇²u − u v² + F(1 − u),  v_t = Dv∇²v + u v² − (F + k) v
 * with Du = 2·10⁻⁵, Dv = 10⁻⁵ (the classic domain length is 2.5).
 */
export const grayScott: SpectralModel = {
  id: 'gray-scott',
  name: 'Gray–Scott',
  description: 'Autocatalytic reaction–diffusion — spots, worms, mitosis and chaos depending on feed F and kill k (shows v).',
  fields: 2,
  params: [
    { key: 'F', label: 'feed F', min: 0.01, max: 0.08, step: 0.0005, default: 0.037 },
    { key: 'k', label: 'kill k', min: 0.04, max: 0.07, step: 0.0005, default: 0.06 },
  ],
  domain: { default: 2.5, min: 0.5, max: 8, step: 0.1 },
  dt: { default: 1, min: 0.1, max: 8, step: 0.1 },
  stepsPerFrame: 8,
  linearSymbol: (k2, f, p) => (f === 0 ? -DU * k2 - p.F! : -DV * k2 - (p.F! + p.k!)),
  nonlinearWgsl: `
    let r = u[0].x * u[1].x * u[1].x;
    if (f == 0u) {
      return vec2<f32>(-r + params.F, 0.0);
    }
    return vec2<f32>(r, 0.0);
  `,
  views: [
    { label: 'v', field: 1, kind: 're' },
    { label: 'u', field: 0, kind: 're' },
  ],
  initial: ({ nx, ny }) => {
    const cells = nx * ny
    const out = new Float32Array(4 * cells)
    for (let i = 0; i < cells; i++) {
      out[2 * i] = 1
      out[2 * (cells + i)] = 0.001 * Math.abs(gaussian())
    }
    // a few perturbed squares seed the pattern
    const half = Math.max(2, Math.round(0.05 * nx))
    for (let s = 0; s < 6; s++) {
      const cx = Math.floor(Math.random() * nx)
      const cy = Math.floor(Math.random() * ny)
      for (let dy = -half; dy <= half; dy++) {
        for (let dx = -half; dx <= half; dx++) {
          const i = ((cy + dy + ny) % ny) * nx + ((cx + dx + nx) % nx)
          out[2 * i] = 0.5
          out[2 * (cells + i)] = 0.25
        }
      }
    }
    return out
  },
}
