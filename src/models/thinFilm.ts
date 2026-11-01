import { defineModel } from './defineModel'
import { noisyState } from './initial'

/** Stabilization: the linear part uses the mobility STAB·h₀³, which exceeds h³ in the droplets (h ≲ 2h₀). */
const STAB = 8

// Disjoining-pressure potential f'(h) = h⁻³ − h⁻⁶ (precursor film at h = 1) and its derivatives
const fpp = (h: number) => -3 / h ** 4 + 6 / h ** 7

/**
 * Thin-film (lubrication) equation with a wetting potential f(h):
 *   ∂t h = ∇·( h³ ∇p ),   p = −∇²h + f'(h),   f'(h) = h⁻³ − h⁻⁶.
 * Flat films thicker than h = 2^(1/3) ≈ 1.26 are linearly unstable (f'' < 0) and dewet into droplets
 * on a precursor film of thickness 1; ∫h is conserved.
 *
 * Expanded: ∇·(M∇p) = M∇²p + M'∇h·∇p with M = h³,
 *   ∇²p = −∇⁴h + f''∇²h + f'''|∇h|²,   ∇p = −∇∇²h + f''∇h.
 * The linear part is  L = −STAB·h₀³ k⁴ − h₀³ f''(h₀) k²  (the exact dispersion relation about the flat
 * film for STAB = 1); the pointwise remainder N is the full right-hand side minus the physical-space
 * image of L. The stiff −k⁴ part is thus treated exactly, as in the Cahn–Hilliard model.
 */
export const thinFilm = defineModel(
  {
    id: 'thin-film',
    name: 'Thin film',
    description: '∂t h = ∇·(h³∇p), p = −∇²h + h⁻³ − h⁻⁶ — dewetting of a liquid film into droplets on a precursor layer; thickness is conserved.',
    fields: 1,
    params: [
      { key: 'h0', label: 'mean thickness h₀', min: 1.1, max: 4, step: 0.01, default: 2, resetOnChange: true },
      { key: 'noise', label: 'initial noise', min: 0.001, max: 0.2, step: 0.001, default: 0.01, resetOnChange: true },
    ],
    domain: { default: 150, min: 30, max: 400, step: 1 },
    dt: { default: 1, min: 0.05, max: 4, step: 0.05 },
    stepsPerFrame: 4,
    paint: { amounts: [0.5], sigma: 0.03 },
    views: [{ label: 'h', field: 0, kind: 're' }],
    presets: [
      {
        name: 'Spinodal dewetting',
        description: 'h₀ = 2: the film breaks up into droplets of wavelength ≈ 24, which then slowly coarsen.',
        params: { h0: 2, noise: 0.01 },
        domain: 150,
        dt: 1,
      },
      {
        name: 'Thin film, short wavelength',
        description: 'h₀ = 1.5 is closer to the stability limit, so the instability is faster and the wavelength ≈ 18 shorter.',
        params: { h0: 1.5, noise: 0.01 },
        domain: 100,
        dt: 1,
      },
    ],
  },
  {
    linearSymbol: (k2, _f, p) => {
      const m0 = p.h0! ** 3
      return -STAB * m0 * k2 * k2 - m0 * fpp(p.h0!) * k2
    },
    usesHigherDerivatives: true,
    nonlinearWgsl: `
      // h is clamped so that a stray negative value cannot make the potential blow up
      let h = max(u[0].x, 0.3);
      let h2 = h * h;
      let m = h2 * h;
      let m0 = params.h0 * params.h0 * params.h0;
      let h04 = params.h0 * params.h0 * params.h0 * params.h0;
      let f2_0 = -3.0 / h04 + 6.0 / (h04 * m0);
      let hm4 = 1.0 / (h2 * h2);
      let f2 = -3.0 * hm4 + 6.0 * hm4 / m;
      let f3 = 12.0 * hm4 / h - 42.0 * hm4 * hm4;
      let hx = ux[0].x;
      let hy = uy[0].x;
      let n = (${STAB}.0 * m0 - m) * bilap[0].x
        + (m * f2 - m0 * f2_0) * lap[0].x
        + m * f3 * (hx * hx + hy * hy)
        + 3.0 * h2 * (hx * (f2 * hx - lapx[0].x) + hy * (f2 * hy - lapy[0].x));
      return vec2<f32>(n, 0.0);
    `,
    initial: (ctx) => noisyState(ctx, [ctx.params.h0 ?? 2], ctx.params.noise ?? 0.01),
  },
)
