// Batched 1D complex FFT: one workgroup per row, whole row held in workgroup memory.
// In-place radix-2 decimation-in-time with a bit-reversed load. Twiddles come from a
// precomputed f64→f32 table (more accurate than evaluating sin/cos in f32).

struct Params {
  n: u32,
  log2n: u32,
  rows: u32,
  inverse: u32,
  scale: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
// tw[k] = exp(-2πi k / n), k in [0, n/2)
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>;

const WG: u32 = 256u;
var<workgroup> buf: array<vec2<f32>, 2048>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = p.n;
  let base = wid.x * n;
  let shift = 32u - p.log2n;

  for (var i = li; i < n; i += WG) {
    buf[reverseBits(i) >> shift] = src[base + i];
  }
  workgroupBarrier();

  // Forward uses the table as is (e^{-i...}); inverse conjugates it.
  let flip = select(-1.0, 1.0, p.inverse == 0u);
  let halfN = n / 2u;
  for (var s = 0u; s < p.log2n; s++) {
    let h = 1u << s;
    for (var t = li; t < halfN; t += WG) {
      let k = t & (h - 1u);
      let i0 = ((t >> s) << (s + 1u)) + k;
      let i1 = i0 + h;
      var w = tw[k << (p.log2n - 1u - s)];
      w.y = w.y * flip;
      let a = buf[i0];
      let b = cmul(w, buf[i1]);
      buf[i0] = a + b;
      buf[i1] = a - b;
    }
    workgroupBarrier();
  }

  for (var i = li; i < n; i += WG) {
    dst[base + i] = buf[i] * p.scale;
  }
}
