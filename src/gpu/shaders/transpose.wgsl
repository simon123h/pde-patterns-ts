// Tiled out-of-place transpose of a complex matrix: src is rows × cols (row-major),
// dst is cols × rows.

struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

// 17 columns: padding avoids shared-memory bank conflicts on the transposed read.
var<workgroup> tile: array<array<vec2<f32>, 17>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let c = wid.x * 16u + lid.x;
  let r = wid.y * 16u + lid.y;
  if (r < p.rows && c < p.cols) {
    tile[lid.y][lid.x] = src[r * p.cols + c];
  }
  workgroupBarrier();

  let orow = wid.x * 16u + lid.y;
  let ocol = wid.y * 16u + lid.x;
  if (orow < p.cols && ocol < p.rows) {
    dst[orow * p.rows + ocol] = tile[lid.x][lid.y];
  }
}
