// Commits the stage result to the state if the controller accepted the step.

struct Params {
  total: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ctl: array<f32>;
@group(0) @binding(2) var<storage, read> next: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> uh: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < p.total && ctl[2] > 0.5) {
    uh[i] = next[i];
  }
}
