/**
 * φ-function coefficients for exponential integrators, evaluated in f64 on the CPU
 * (the GPU only has f32, where these suffer catastrophic cancellation for small |z|).
 *
 *   φ1(z) = (e^z − 1) / z
 *   φ2(z) = (e^z − 1 − z) / z²
 *   φ3(z) = (e^z − 1 − z − z²/2) / z³
 *
 * Evaluated with the Kassam–Trefethen contour-integral trick: average over a circle
 * of radius R around z in the complex plane, which is stable for any z.
 */
export interface Phi {
  phi1: number
  phi2: number
  phi3: number
}

const CONTOUR_POINTS = 32
const CONTOUR_RADIUS = 1

/** Contour-integral evaluation of φ1..φ3 at real z; accurate everywhere but ~100× slower than `phiFunctions`. */
export function phiContour(z: number): Phi {
  let p1 = 0
  let p2 = 0
  let p3 = 0
  for (let j = 0; j < CONTOUR_POINTS; j++) {
    const theta = (2 * Math.PI * (j + 0.5)) / CONTOUR_POINTS
    const zr = z + CONTOUR_RADIUS * Math.cos(theta)
    const zi = CONTOUR_RADIUS * Math.sin(theta)
    // e^{zr + i zi}
    const ex = Math.exp(zr)
    const er = ex * Math.cos(zi)
    const ei = ex * Math.sin(zi)
    // 1/z, 1/z², 1/z³ for complex z
    const d = zr * zr + zi * zi
    const ir = zr / d
    const ii = -zi / d
    const i2r = ir * ir - ii * ii
    const i2i = 2 * ir * ii
    const i3r = i2r * ir - i2i * ii
    const i3i = i2r * ii + i2i * ir
    // φ1 = (e^z − 1)/z
    p1 += (er - 1) * ir - ei * ii
    // φ2 = (e^z − 1 − z)/z²
    p2 += (er - 1 - zr) * i2r - (ei - zi) * i2i
    // φ3 = (e^z − 1 − z − z²/2)/z³
    const z2r = zr * zr - zi * zi
    const z2i = 2 * zr * zi
    p3 += (er - 1 - zr - z2r / 2) * i3r - (ei - zi - z2i / 2) * i3i
  }
  return { phi1: p1 / CONTOUR_POINTS, phi2: p2 / CONTOUR_POINTS, phi3: p3 / CONTOUR_POINTS }
}

/**
 * φ1..φ3 at real z (the linear symbol times the step size), cheap enough to evaluate for
 * millions of modes: Taylor series for |z| < 1, closed forms (no cancellation problem) beyond.
 */
export function phiFunctions(z: number): Phi {
  if (Math.abs(z) >= 1) {
    const em1 = Math.expm1(z)
    return {
      phi1: em1 / z,
      phi2: (em1 - z) / (z * z),
      phi3: (em1 - z - (z * z) / 2) / (z * z * z),
    }
  }
  // φ_j(z) = Σ_k z^k / (k + j)!   (20 terms: remainder < 1e-18 for |z| < 1)
  let term1 = 1 // 1/1!
  let term2 = 1 / 2
  let term3 = 1 / 6
  let p1 = term1
  let p2 = term2
  let p3 = term3
  for (let k = 1; k < 20; k++) {
    term1 *= z / (k + 1)
    term2 *= z / (k + 2)
    term3 *= z / (k + 3)
    p1 += term1
    p2 += term2
    p3 += term3
  }
  return { phi1: p1, phi2: p2, phi3: p3 }
}

/**
 * Coefficients of the Krogstad ETDRK4 scheme for one mode with linear symbol L and step h:
 *
 *   a = E2·u + A1·N(u)
 *   b = E2·u + A1·N(u) + B2·(N(a) − N(u))
 *   c = E·u  + C1·N(u) + C2·(N(b) − N(u))
 *   u⁺ = E·u + F1·N(u) + F2·(N(a) + N(b)) + F3·N(c)
 *
 * with E = e^{hL}, E2 = e^{hL/2}. The A1…F3 factors multiply nonlinear terms; callers zero
 * them for dealiased modes.
 */
export interface KrogstadCoefficients {
  E: number
  E2: number
  A1: number
  B2: number
  C1: number
  C2: number
  F1: number
  F2: number
  F3: number
}

export function krogstadCoefficients(L: number, h: number): KrogstadCoefficients {
  const z = h * L
  const full = phiFunctions(z)
  const half = phiFunctions(z / 2)
  return {
    E: Math.exp(z),
    E2: Math.exp(z / 2),
    A1: (h / 2) * half.phi1,
    B2: h * half.phi2,
    C1: h * full.phi1,
    C2: 2 * h * full.phi2,
    F1: h * (full.phi1 - 3 * full.phi2 + 4 * full.phi3),
    F2: 2 * h * (full.phi2 - 2 * full.phi3),
    F3: h * (4 * full.phi3 - full.phi2),
  }
}

/** One Krogstad ETDRK4 step for a scalar ODE u' = L u + N(u); reference for tests. */
export function krogstadStep(u: number, L: number, h: number, N: (u: number) => number): number {
  const c = krogstadCoefficients(L, h)
  const Nn = N(u)
  const a = c.E2 * u + c.A1 * Nn
  const Na = N(a)
  const b = c.E2 * u + c.A1 * Nn + c.B2 * (Na - Nn)
  const Nb = N(b)
  const cc = c.E * u + c.C1 * Nn + c.C2 * (Nb - Nn)
  const Nc = N(cc)
  return c.E * u + c.F1 * Nn + c.F2 * (Na + Nb) + c.F3 * Nc
}
