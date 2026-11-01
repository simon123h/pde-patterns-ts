// Adds Gaussian brush stamps to the spectral state: the Fourier transform of
// A·exp(−|x − x_s|²/2σ²) is A·2πσ²/dx² · exp(−σ²k²/2) · e^{−i k·x_s} (unnormalized forward FFT).

struct Params {
  count: u32,
  nx: u32,
  ny: u32,
  modes: u32, // spectral modes per field
  total: u32,
  sigma: f32, // brush radius in domain units
  dx: f32,
  pad0: u32,
  amounts: vec4<f32>, // per-field amplitude
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> stamps: array<vec4<f32>>; // (x, y, scale, unused)
@group(0) @binding(2) var<storage, read_write> uh: array<vec2<f32>>;

const TAU: f32 = 6.283185307179586;

fn wavenumber(i: u32, n: u32) -> f32 {
  var m = f32(i);
  if (i * 2u >= n) {
    m = f32(i) - f32(n);
  }
  return TAU * m / (f32(n) * p.dx);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= p.total) {
    return;
  }
  let field = g / p.modes;
  let r = g % p.modes;
  let kx = wavenumber(r / p.ny, p.nx);
  let ky = wavenumber(r % p.ny, p.ny);
  let k2 = kx * kx + ky * ky;
  let envelope = TAU * p.sigma * p.sigma / (p.dx * p.dx) * exp(-0.5 * p.sigma * p.sigma * k2);

  var sum = vec2<f32>(0.0);
  for (var s = 0u; s < p.count; s++) {
    let st = stamps[s];
    let phase = -(kx * st.x + ky * st.y);
    sum += st.z * vec2<f32>(cos(phase), sin(phase));
  }
  uh[g] += p.amounts[field] * envelope * sum;
}
