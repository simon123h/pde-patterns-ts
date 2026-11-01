import { noisyState } from './initial'
import { defineModel } from './defineModel'

/** ∂t u = r u − (1 + ∇²)² u + g u² − u³ */
export const swiftHohenberg = defineModel(
  {
    id: 'swift-hohenberg',
    name: 'Swift–Hohenberg',
    description: '∂t u = r u − (1+∇²)² u + g u² − u³ — stripes, hexagons and labyrinths of wavelength 2π.',
    fields: 1,
    params: [
      { key: 'r', label: 'r', min: -1, max: 1, step: 0.01, default: 0.2 },
      { key: 'g', label: 'g (quadratic)', min: -2, max: 2, step: 0.01, default: 0 },
    ],
    domain: { default: 128, min: 20, max: 400, step: 1 },
    dt: { default: 0.2, min: 0.01, max: 2, step: 0.01 },
    stepsPerFrame: 2,
    paint: { amounts: [0.5], sigma: 0.03 },
    views: [{ label: 'u', field: 0, kind: 're' }],
    presets: [
      { name: 'Stripes', description: 'Symmetric nonlinearity: rolls with wavelength 2π.', params: { r: 0.2, g: 0 } },
      { name: 'Hexagons', description: 'The quadratic term favours hexagonal spots.', params: { r: 0.2, g: 1 } },
    ],
  },
  {
    // In Fourier space ∇² → −k², so L = r − (1 − k²)²
    linearSymbol: (k2, _f, p) => p.r! - (1 - k2) ** 2,
    nonlinearWgsl: 'return vec2<f32>(params.g * u[0].x * u[0].x - u[0].x * u[0].x * u[0].x, 0.0);',
    initial: (ctx) => noisyState(ctx, [0], 0.1),
  },
)
