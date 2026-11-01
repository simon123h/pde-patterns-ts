import type { SpectralModel } from '../models/types'

/** Generates the pointwise-nonlinearity kernel: physN[f] = N_f(phys) for all fields. */
export function nonlinearShader(model: SpectralModel, cells: number): string {
  const fields = model.fields
  const struct = model.params.length
    ? model.params.map((p) => `  ${p.key}: f32,`).join('\n')
    : '  unused: f32,'
  return /* wgsl */ `
struct Params {
${struct}
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> phys: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> physN: array<vec2<f32>>;

const NF: u32 = ${fields}u;
const CELLS: u32 = ${cells}u;

fn nonlinear(u: array<f32, ${fields}>, f: u32) -> f32 {
  ${model.nonlinearWgsl}
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = params;
  let i = gid.x;
  if (i >= CELLS) {
    return;
  }
  var u: array<f32, ${fields}>;
  for (var f = 0u; f < NF; f++) {
    u[f] = phys[f * CELLS + i].x;
  }
  for (var f = 0u; f < NF; f++) {
    physN[f * CELLS + i] = vec2<f32>(nonlinear(u, f), 0.0);
  }
}
`
}
