// Fullscreen render of a periodic scalar field with bilinear sampling and a colormap.

struct Params {
  nx: u32,
  ny: u32,
  offset: u32,
  colormap: u32,
  width: f32,
  height: f32,
  kind: u32, // 0 Re, 1 Im, 2 |.|, 3 arg
  diverging: u32, // 1: symmetric range centered on zero
  real: u32, // 1: field is plain f32 instead of interleaved complex
  pad2: u32,
  pad3: u32,
  pad4: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> range: array<f32, 2>;
// 256-entry RGBA tables, one per colormap
@group(0) @binding(3) var<storage, read> palette: array<vec4<f32>>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vi & 2u) * 2.0 - 1.0;
  return vec4<f32>(x, y, 0.0, 1.0);
}

fn texel(ix: i32, iy: i32) -> vec2<f32> {
  let x = u32(((ix % i32(p.nx)) + i32(p.nx)) % i32(p.nx));
  let y = u32(((iy % i32(p.ny)) + i32(p.ny)) % i32(p.ny));
  let idx = p.offset + y * p.nx + x;
  if (p.real == 1u) {
    return vec2<f32>(field[idx], 0.0);
  }
  return vec2<f32>(field[2u * idx], field[2u * idx + 1u]);
}

fn lookup(map: u32, t: f32) -> vec3<f32> {
  let x = t * 255.0;
  let i = u32(floor(x));
  let j = min(i + 1u, 255u);
  return mix(palette[map * 256u + i].rgb, palette[map * 256u + j].rgb, x - floor(x));
}

fn value(v: vec2<f32>) -> f32 {
  switch (p.kind) {
    case 0u: { return v.x; }
    case 1u: { return v.y; }
    case 2u: { return length(v); }
    default: { return atan2(v.y, v.x); }
  }
}

// Cyclic palette for phases.
fn cyclic(t: f32) -> vec3<f32> {
  return 0.5 + 0.5 * cos(6.283185307 * (t + vec3<f32>(0.0, 0.33333, 0.66667)));
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / vec2<f32>(p.width, p.height);
  let g = uv * vec2<f32>(f32(p.nx), f32(p.ny)) - vec2<f32>(0.5);
  let base = floor(g);
  let f = g - base;
  let ix = i32(base.x);
  let iy = i32(base.y);
  // interpolate the complex value, then reduce (keeps phases continuous across the ±π cut)
  let v = value(mix(
    mix(texel(ix, iy), texel(ix + 1, iy), f.x),
    mix(texel(ix, iy + 1), texel(ix + 1, iy + 1), f.x),
    f.y,
  ));
  var lo = range[0];
  var hi = range[1];
  if (p.kind == 3u) {
    lo = -3.14159265;
    hi = 3.14159265;
  }
  if (p.diverging == 1u) {
    let m = max(abs(lo), abs(hi));
    lo = -m;
    hi = m;
  }
  let t = clamp((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
  var color = lookup(p.colormap, t);
  if (p.kind == 3u) {
    color = cyclic(t);
  }
  return vec4<f32>(color, 1.0);
}
