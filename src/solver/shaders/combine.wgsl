// Pointwise ETDRK4 (Krogstad) stage combination in Fourier space with complex coefficients.
// Coefficient planes (each `total` long): 0 E, 1 E2, 2 A1, 3 B2, 4 C1, 5 C2, 6 F1, 7 F2, 8 F3.

struct Params {
  stage: u32,
  total: u32,
  adapt: u32, // 1: also write per-workgroup error sums in the final stage
  pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> coef: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> uh: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> n0: array<vec2<f32>>; // N(u_n)
@group(0) @binding(4) var<storage, read> n1: array<vec2<f32>>; // N(a)
@group(0) @binding(5) var<storage, read> n2: array<vec2<f32>>; // N(b)
@group(0) @binding(6) var<storage, read> n3: array<vec2<f32>>; // N(c)
@group(0) @binding(7) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> partials: array<vec2<f32>>; // (Σ|u⁺ − c|², Σ|u⁺|²) per workgroup

var<workgroup> redD: array<f32, 256>;
var<workgroup> redU: array<f32, 256>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn c(plane: u32, i: u32) -> vec2<f32> {
  return coef[plane * p.total + i];
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
  let estimate = p.stage == 3u && p.adapt == 1u;
  var d = 0.0;
  var un = 0.0;
  if (i < p.total) {
    let u = uh[i];
    var r: vec2<f32>;
    switch (p.stage) {
      case 0u: { // a
        r = cmul(c(1u, i), u) + cmul(c(2u, i), n0[i]);
      }
      case 1u: { // b
        r = cmul(c(1u, i), u) + cmul(c(2u, i), n0[i]) + cmul(c(3u, i), n1[i] - n0[i]);
      }
      case 2u: { // c
        r = cmul(c(0u, i), u) + cmul(c(4u, i), n0[i]) + cmul(c(5u, i), n2[i] - n0[i]);
      }
      default: { // u_{n+1}
        r = cmul(c(0u, i), u) + cmul(c(6u, i), n0[i]) + cmul(c(7u, i), n1[i] + n2[i]) + cmul(c(8u, i), n3[i]);
      }
    }
    if (estimate) {
      // `out` still holds the third stage c: the gap to the 4th-order result estimates the error
      let diff = r - out[i];
      d = dot(diff, diff);
      un = dot(r, r);
    }
    out[i] = r;
  }
  if (estimate) {
    redD[li] = d;
    redU[li] = un;
    workgroupBarrier();
    var s = 128u;
    loop {
      if (s == 0u) {
        break;
      }
      if (li < s) {
        redD[li] += redD[li + s];
        redU[li] += redU[li + s];
      }
      workgroupBarrier();
      s = s >> 1u;
    }
    if (li == 0u) {
      partials[wid.x] = vec2<f32>(redD[0], redU[0]);
    }
  }
}
