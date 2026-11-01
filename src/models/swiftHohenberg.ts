import type { SpectralModel } from './types'

/** ∂t u = r u − (1 + ∇²)² u + g u² − u³ */
export const swiftHohenberg: SpectralModel = {
  id: 'swift-hohenberg',
  name: 'Swift–Hohenberg',
  fields: 1,
  params: [
    { key: 'r', label: 'r', min: -1, max: 1, step: 0.01, default: 0.2 },
    { key: 'g', label: 'g (quadratic)', min: -2, max: 2, step: 0.01, default: 0 },
  ],
  // In Fourier space ∇² → −k², so L = r − (1 − k²)²
  linearSymbol: (k2, _field, p) => p.r! - (1 - k2) ** 2,
  nonlinearWgsl: 'return params.g * u[0] * u[0] - u[0] * u[0] * u[0];',
  noiseAmplitude: 0.1,
}
