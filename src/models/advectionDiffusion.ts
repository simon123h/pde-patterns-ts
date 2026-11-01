import type { SpectralModel } from './types'

/**
 * Convection–diffusion of a passive scalar:  ∂t u + v·∇u = D ∇²u
 * with a uniform drift c = (cx, cy) plus an optional cellular (Taylor–Green) flow
 *   v = A (sin κx cos κy, −cos κx sin κy),  κ = 2π n / L  (n × n counter-rotating vortices).
 * The drift is exact in the linear operator; the divergence-free cellular flow is advanced
 * pointwise (−v·∇u) like a nonlinearity, so ∫u is conserved.
 */
export const advectionDiffusion: SpectralModel = {
  id: 'advection-diffusion',
  name: 'Convection–diffusion',
  description: '∂t u + v·∇u = D∇²u — a scalar stirred by a uniform drift and an n×n vortex flow; stripes get stretched into filaments until diffusion smooths them.',
  fields: 1,
  params: [
    { key: 'D', label: 'diffusivity D', min: 0.0005, max: 0.2, step: 0.0005, default: 0.005 },
    { key: 'cx', label: 'drift cₓ', min: -3, max: 3, step: 0.01, default: 0.3 },
    { key: 'cy', label: 'drift c_y', min: -3, max: 3, step: 0.01, default: 0 },
    { key: 'A', label: 'vortex flow amplitude A', min: 0, max: 3, step: 0.01, default: 1 },
    { key: 'n', label: 'vortices per side n', min: 1, max: 6, step: 1, default: 2 },
  ],
  domain: { default: 6.28, min: 2, max: 20, step: 0.01 },
  dt: { default: 0.01, min: 0.001, max: 0.05, step: 0.001 },
  stepsPerFrame: 4,
  // û_t = −i (c·k) û − D k² û
  linearSymbol: (k2, _f, p) => -p.D! * k2,
  linearSymbolImag: (_k2, _f, p, kx, ky) => -(p.cx! * kx + p.cy! * ky),
  usesGradient: true,
  nonlinearWgsl: `
    let kappa = 6.283185307 * params.n / (params.dx * f32(NX));
    let vx = params.A * sin(kappa * xy.x) * cos(kappa * xy.y);
    let vy = -params.A * cos(kappa * xy.x) * sin(kappa * xy.y);
    return vec2<f32>(-(vx * ux[0].x + vy * uy[0].x), 0.0);
  `,
  views: [{ label: 'u', field: 0, kind: 're' }],
  // smooth periodic square wave of stripes along x, so the vortex flow has something to stretch
  initial: ({ nx, ny }) => {
    const out = new Float32Array(2 * nx * ny)
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) out[2 * (y * nx + x)] = Math.tanh(3 * Math.sin((2 * Math.PI * x) / nx))
    }
    return out
  },
}
