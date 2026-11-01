// Spectral gradient: gx = i·kx·û, gy = i·ky·û for every field (Nyquist derivative set to zero).

struct Params {
  nx: u32,
  ny: u32,
  cells: u32,
  total: u32,
  dx: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> uh: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> gx: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> gy: array<vec2<f32>>;

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
  let r = g % p.cells;
  // spectral layout is transposed: index = kx * ny + ky
  let kx = wavenumber(r / p.ny, p.nx, p.dx);
  let ky = wavenumber(r % p.ny, p.ny, p.dx);
  let u = uh[g];
  gx[g] = vec2<f32>(-kx * u.y, kx * u.x);
  gy[g] = vec2<f32>(-ky * u.y, ky * u.x);
}
