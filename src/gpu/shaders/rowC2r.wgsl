// Batched complex-to-real row FFT, the inverse of rowR2c.wgsl: a half spectrum X_0 … X_{n/2}
// is folded into the n/2-point complex row Z_k = E_k + i·O_k with
//   E_k = (X_k + conj X_{h−k})/2,  O_k = (X_k − conj X_{h−k})·conj(W_n^k)/2,
// inverse-transformed, and unpacked to x_{2j} = Re z_j, x_{2j+1} = Im z_j.
// The imaginary parts of X_0 and X_{n/2} are ignored (they vanish for a real signal).

struct Params {
  n: u32, // real row length
  logh: u32, // log2(n / 2)
  rows: u32,
  pad0: u32,
  scale: f32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>; // exp(−2πi k / h), k < h/2
@group(0) @binding(4) var<storage, read> wn: array<vec2<f32>>; // exp(−2πi k / n), k = 0 … h

const WG: u32 = 256u;
var<workgroup> buf: array<vec2<f32>, 2048>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn conj(a: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x, -a.y);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = p.n / 2u;
  let srcBase = wid.x * (h + 1u);
  let dstBase = wid.x * p.n;
  let shift = 32u - p.logh;

  for (var k = li; k < h; k += WG) {
    var xk = src[srcBase + k];
    var xn = src[srcBase + h - k];
    if (k == 0u) {
      xk.y = 0.0;
      xn.y = 0.0;
    }
    let e = 0.5 * (xk + conj(xn));
    let o = 0.5 * cmul(xk - conj(xn), conj(wn[k]));
    buf[reverseBits(k) >> shift] = e + vec2<f32>(-o.y, o.x); // E + i·O
  }
  workgroupBarrier();

  let halfH = h / 2u;
  for (var s = 0u; s < p.logh; s++) {
    let m = 1u << s;
    for (var t = li; t < halfH; t += WG) {
      let k = t & (m - 1u);
      let i0 = ((t >> s) << (s + 1u)) + k;
      let i1 = i0 + m;
      let w = conj(tw[k << (p.logh - 1u - s)]); // inverse transform
      let a = buf[i0];
      let b = cmul(w, buf[i1]);
      buf[i0] = a + b;
      buf[i1] = a - b;
    }
    workgroupBarrier();
  }

  for (var j = li; j < h; j += WG) {
    let z = buf[j] * p.scale;
    dst[dstBase + 2u * j] = z.x;
    dst[dstBase + 2u * j + 1u] = z.y;
  }
}
