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

/** φ1..φ3 at real z (the linear symbol times the step size; z ≤ 0 for dissipative modes). */
export function phiFunctions(z: number): Phi {
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
