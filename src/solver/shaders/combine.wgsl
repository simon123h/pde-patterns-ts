// Pointwise ETDRK4 (Krogstad) stage combination in Fourier space.
// Coefficient planes (each `total` long): 0 E, 1 E2, 2 A1, 3 B2, 4 C1, 5 C2, 6 F1, 7 F2, 8 F3.

struct Params {
  stage: u32,
  total: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> coef: array<f32>;
@group(0) @binding(2) var<storage, read> uh: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> n0: array<vec2<f32>>; // N(u_n)
@group(0) @binding(4) var<storage, read> n1: array<vec2<f32>>; // N(a)
@group(0) @binding(5) var<storage, read> n2: array<vec2<f32>>; // N(b)
@group(0) @binding(6) var<storage, read> n3: array<vec2<f32>>; // N(c)
@group(0) @binding(7) var<storage, read_write> out: array<vec2<f32>>;

fn c(plane: u32, i: u32) -> f32 {
  return coef[plane * p.total + i];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.total) {
    return;
  }
  let u = uh[i];
  var r: vec2<f32>;
  switch (p.stage) {
    case 0u: { // a
      r = c(1u, i) * u + c(2u, i) * n0[i];
    }
    case 1u: { // b
      r = c(1u, i) * u + c(2u, i) * n0[i] + c(3u, i) * (n1[i] - n0[i]);
    }
    case 2u: { // c
      r = c(0u, i) * u + c(4u, i) * n0[i] + c(5u, i) * (n2[i] - n0[i]);
    }
    default: { // u_{n+1}
      r = c(0u, i) * u + c(6u, i) * n0[i] + c(7u, i) * (n1[i] + n2[i]) + c(8u, i) * n3[i];
    }
  }
  out[i] = r;
}
