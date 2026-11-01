// GPU evaluation of the Krogstad ETDRK4 coefficients for the current step size ctl[0], used by
// adaptive time stepping where h changes too often for CPU-side (f64) coefficient builds.
// φ1..φ3 of complex z: Taylor series for |z| < 1, recurrence φ_{j+1} = (φ_j − 1/j!)/z beyond.
// Output planes (each `total` long): E, E2, A1, B2, C1, C2, F1, F2, F3.

struct Params {
  total: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> sym: array<vec4<f32>>; // (Re L, Im L, nonlinear multiplier, 0)
@group(0) @binding(2) var<storage, read> ctl: array<f32>;
@group(0) @binding(3) var<storage, read_write> coef: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let d = dot(b, b);
  return vec2<f32>((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

// sin/cos with explicit range reduction (Cody–Waite, three-part π/2) and short polynomials on
// [−π/4, π/4]: accurate to ~1e-7 everywhere, whereas the built-in sin/cos may only be good to
// about 2⁻¹¹ (and the phase of e^{iθ} for large θ matters for dispersive terms).
fn sincos(x: f32) -> vec2<f32> {
  let q = round(x * 0.6366197723675814);
  var r = x - q * 1.5703125;
  r = r - q * 4.837512969970703125e-4;
  r = r - q * 7.549789948768648e-8;
  let r2 = r * r;
  let s = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362880.0)))));
  let c = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40320.0 + r2 * (-1.0 / 3628800.0)))));
  switch (u32(i32(q) & 3)) {
    case 0u: { return vec2<f32>(s, c); }
    case 1u: { return vec2<f32>(c, -s); }
    case 2u: { return vec2<f32>(-s, -c); }
    default: { return vec2<f32>(-c, s); }
  }
}

fn cexp(z: vec2<f32>) -> vec2<f32> {
  let e = exp(z.x);
  let sc = sincos(z.y);
  return vec2<f32>(e * sc.y, e * sc.x);
}

struct Phi {
  p1: vec2<f32>,
  p2: vec2<f32>,
  p3: vec2<f32>,
}

fn phi(z: vec2<f32>) -> Phi {
  var r: Phi;
  if (dot(z, z) >= 1.0) {
    r.p1 = cdiv(cexp(z) - vec2<f32>(1.0, 0.0), z);
    r.p2 = cdiv(r.p1 - vec2<f32>(1.0, 0.0), z);
    r.p3 = cdiv(r.p2 - vec2<f32>(0.5, 0.0), z);
  } else {
    var t1 = vec2<f32>(1.0, 0.0);
    var t2 = vec2<f32>(0.5, 0.0);
    var t3 = vec2<f32>(1.0 / 6.0, 0.0);
    r.p1 = t1;
    r.p2 = t2;
    r.p3 = t3;
    for (var k = 1; k < 14; k++) {
      let kf = f32(k);
      t1 = cmul(t1, z) / (kf + 1.0);
      t2 = cmul(t2, z) / (kf + 2.0);
      t3 = cmul(t3, z) / (kf + 3.0);
      r.p1 += t1;
      r.p2 += t2;
      r.p3 += t3;
    }
  }
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= p.total) {
    return;
  }
  let s = sym[g];
  let h = ctl[0];
  let z = h * s.xy;
  let full = phi(z);
  let half = phi(0.5 * z);
  let m = s.z;
  let t = p.total;
  coef[g] = cexp(z);
  coef[t + g] = cexp(0.5 * z);
  coef[2u * t + g] = (0.5 * h * m) * half.p1;
  coef[3u * t + g] = (h * m) * half.p2;
  coef[4u * t + g] = (h * m) * full.p1;
  coef[5u * t + g] = (2.0 * h * m) * full.p2;
  coef[6u * t + g] = (h * m) * (full.p1 - 3.0 * full.p2 + 4.0 * full.p3);
  coef[7u * t + g] = (2.0 * h * m) * (full.p2 - 2.0 * full.p3);
  coef[8u * t + g] = (h * m) * (4.0 * full.p3 - full.p2);
}
