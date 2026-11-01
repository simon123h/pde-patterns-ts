// Smoothed min/max of one field, computed by a single workgroup on a strided subsample.
// The result stays on the GPU; the render shader reads it directly (no CPU readback).

struct Params {
  count: u32,
  offset: u32,
  stride: u32,
  first: u32,
  blend: f32,
  kind: u32,
  real: u32, // 1: field is plain f32 instead of interleaved complex
  pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read_write> range: array<f32, 2>;

// kind: 0 Re, 1 Im, 2 |.|, 3 arg
fn value(v: vec2<f32>, kind: u32) -> f32 {
  switch (kind) {
    case 0u: { return v.x; }
    case 1u: { return v.y; }
    case 2u: { return length(v); }
    default: { return atan2(v.y, v.x); }
  }
}

var<workgroup> mins: array<f32, 256>;
var<workgroup> maxs: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var lo = 3.0e38;
  var hi = -3.0e38;
  var i = li * p.stride;
  loop {
    if (i >= p.count) {
      break;
    }
    let idx = p.offset + i;
    var z = vec2<f32>(0.0);
    if (p.real == 1u) {
      z = vec2<f32>(field[idx], 0.0);
    } else {
      z = vec2<f32>(field[2u * idx], field[2u * idx + 1u]);
    }
    let v = value(z, p.kind);
    lo = min(lo, v);
    hi = max(hi, v);
    i += 256u * p.stride;
  }
  mins[li] = lo;
  maxs[li] = hi;
  workgroupBarrier();

  var s = 128u;
  loop {
    if (s == 0u) {
      break;
    }
    if (li < s) {
      mins[li] = min(mins[li], mins[li + s]);
      maxs[li] = max(maxs[li], maxs[li + s]);
    }
    workgroupBarrier();
    s = s >> 1u;
  }

  if (li == 0u) {
    if (p.first == 1u) {
      range[0] = mins[0];
      range[1] = maxs[0];
    } else {
      range[0] = mix(range[0], mins[0], p.blend);
      range[1] = mix(range[1], maxs[0], p.blend);
    }
  }
}
