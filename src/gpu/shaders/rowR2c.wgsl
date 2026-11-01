// Batched real-to-complex row FFT. A real row of n samples is transformed as a complex row of
// n/2 samples z_j = x_{2j} + i·x_{2j+1}; the half spectrum X_0 … X_{n/2} follows from
//   E_k = (Z_k + conj Z_{h−k})/2,  O_k = (Z_k − conj Z_{h−k})/(2i),  X_k = E_k + W_n^k O_k.
// One workgroup per row, the whole half-length row lives in workgroup memory.

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
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
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
  let srcBase = wid.x * p.n;
  let dstBase = wid.x * (h + 1u);
  let shift = 32u - p.logh;

  for (var j = li; j < h; j += WG) {
    buf[reverseBits(j) >> shift] = vec2<f32>(src[srcBase + 2u * j], src[srcBase + 2u * j + 1u]);
  }
  workgroupBarrier();

  let halfH = h / 2u;
  for (var s = 0u; s < p.logh; s++) {
    let m = 1u << s;
    for (var t = li; t < halfH; t += WG) {
      let k = t & (m - 1u);
      let i0 = ((t >> s) << (s + 1u)) + k;
      let i1 = i0 + m;
      let w = tw[k << (p.logh - 1u - s)];
      let a = buf[i0];
      let b = cmul(w, buf[i1]);
      buf[i0] = a + b;
      buf[i1] = a - b;
    }
    workgroupBarrier();
  }

  for (var k = li; k <= h; k += WG) {
    let zk = buf[k % h];
    let zn = conj(buf[(h - k) % h]);
    let e = 0.5 * (zk + zn);
    let d = zk - zn;
    let o = 0.5 * vec2<f32>(d.y, -d.x); // d / (2i)
    dst[dstBase + k] = (e + cmul(wn[k], o)) * p.scale;
  }
}
