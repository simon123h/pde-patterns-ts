import { noisyState } from './initial'
import { defineModel } from './defineModel'

/** ∂t u = −∇²u − ν∇⁴u − ½ |∇u|² (2D Kuramoto–Sivashinsky) */
export const kuramotoSivashinsky = defineModel(
  {
    id: 'kuramoto-sivashinsky',
    name: 'Kuramoto–Sivashinsky',
    description: '∂t u = −∇²u − ν∇⁴u − ½|∇u|² — spatiotemporal chaos with a characteristic cell size.',
    fields: 1,
    params: [{ key: 'nu', label: 'ν (4th-order damping)', min: 0.3, max: 3, step: 0.01, default: 1 }],
    domain: { default: 100, min: 20, max: 400, step: 1 },
    dt: { default: 0.1, min: 0.01, max: 1, step: 0.01 },
    stepsPerFrame: 3,
    paint: { amounts: [1], sigma: 0.03 },
    views: [{ label: 'u', field: 0, kind: 're' }],
    // the field grows to large amplitudes (rms of tens) and the 2D scalar form can blow up independently of dt,
    // so rolling back and halving dt does not help; the blow-up check is off for this equation
    blowupRecovery: false,
  },
  {
    // ∇² → −k², ∇⁴ → k⁴
    linearSymbol: (k2, _f, p) => k2 - p.nu! * k2 * k2,
    // The equation only involves derivatives of u, so u + c is a solution too, and −½|∇u|² makes the mean fall
    // without bound (in f32 the fluctuations are eventually lost against it). Freezing the k = 0 mode of the
    // nonlinearity is the same as subtracting the mean drift ½⟨|∇u|²⟩ and keeps the mean at its initial value.
    nonlinearSymbol: (k2) => (k2 === 0 ? 0 : 1),
    usesGradient: true,
    nonlinearWgsl: 'return vec2<f32>(-0.5 * (ux[0].x * ux[0].x + uy[0].x * uy[0].x), 0.0);',
    initial: (ctx) => noisyState(ctx, [0], 0.1),
  },
)
