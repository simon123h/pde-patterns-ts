/**
 * Spectral derivative kernel: for every field, writes derived spectra of û for the nonlinearity.
 * With `higher` false these are ∂x u and ∂y u; with `higher` true also ∇²u, ∂x∇²u, ∂y∇²u and ∇⁴u
 * (in this order, see `DERIVATIVES`). The Nyquist wavenumber is set to zero.
 */
export const DERIVATIVES = ['ux', 'uy', 'lap', 'lapx', 'lapy', 'bilap'] as const

/** Number of derived fields the solver provides to the nonlinearity. */
export const derivativeCount = (higher: boolean) => (higher ? DERIVATIVES.length : 2)

export function derivativeShader(higher: boolean): string {
  const count = derivativeCount(higher)
  const outputs = Array.from({ length: count }, (_, i) => `@group(0) @binding(${i + 2}) var<storage, read_write> out${i}: array<vec2<f32>>;`).join('\n')
  // multipliers applied to û; `i` is the imaginary unit
  const writes = [
    'out0[g] = vec2<f32>(-kx * u.y, kx * u.x);', // i kx
    'out1[g] = vec2<f32>(-ky * u.y, ky * u.x);', // i ky
    'out2[g] = -k2 * u;', // −k²
    'out3[g] = k2 * vec2<f32>(kx * u.y, -kx * u.x);', // −i kx k²
    'out4[g] = k2 * vec2<f32>(ky * u.y, -ky * u.x);', // −i ky k²
    'out5[g] = k2 * k2 * u;', // k⁴
  ]
    .slice(0, count)
    .join('\n  ')
  return /* wgsl */ `
struct Params {
  nx: u32,
  ny: u32,
  modes: u32, // spectral modes per field
  total: u32,
  dx: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> uh: array<vec2<f32>>;
${outputs}

const TAU: f32 = 6.283185307179586;

fn wavenumber(i: u32, n: u32, dx: f32) -> f32 {
  if (i * 2u == n) {
    return 0.0;
  }
  var m = f32(i);
  if (i * 2u > n) {
    m = f32(i) - f32(n);
  }
  return TAU * m / (f32(n) * dx);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= p.total) {
    return;
  }
  let r = g % p.modes;
  // spectral layout is transposed: index = kx * ny + ky
  let kx = wavenumber(r / p.ny, p.nx, p.dx);
  let ky = wavenumber(r % p.ny, p.ny, p.dx);
  let k2 = kx * kx + ky * ky;
  let u = uh[g];
  ${writes}
}
`
}
