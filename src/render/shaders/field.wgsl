// Fullscreen render of a periodic scalar field with bilinear sampling and a colormap.

struct Params {
  nx: u32,
  ny: u32,
  offset: u32,
  colormap: u32,
  width: f32,
  height: f32,
  kind: u32, // 0 Re, 1 Im, 2 |.|, 3 arg
  pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> field: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> range: array<f32, 2>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vi & 2u) * 2.0 - 1.0;
  return vec4<f32>(x, y, 0.0, 1.0);
}

fn texel(ix: i32, iy: i32) -> vec2<f32> {
  let x = u32(((ix % i32(p.nx)) + i32(p.nx)) % i32(p.nx));
  let y = u32(((iy % i32(p.ny)) + i32(p.ny)) % i32(p.ny));
  return field[p.offset + y * p.nx + x];
}

// Matt Zucker's polynomial fit of matplotlib's viridis.
fn viridis(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3<f32>(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3<f32>(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3<f32>(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3<f32>(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3<f32>(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3<f32>(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3<f32>(0.0), vec3<f32>(1.0));
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
  let t = clamp((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
  var color = viridis(t);
  if (p.colormap == 1u) {
    color = vec3<f32>(t);
  }
  if (p.kind == 3u) {
    color = cyclic(t);
  }
  return vec4<f32>(color, 1.0);
}
