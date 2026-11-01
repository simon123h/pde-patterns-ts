import type { SpectralModel } from '../models/types'

/** Generates the pointwise-nonlinearity kernel: physN[f] = N_f(phys, ∇phys) for all fields. */
export function nonlinearShader(model: SpectralModel, cells: number): string {
  const fields = model.fields
  const struct = model.params.length
    ? model.params.map((p) => `  ${p.key}: f32,`).join('\n')
    : '  unused: f32,'
  const gradientBindings = model.usesGradient
    ? `@group(0) @binding(3) var<storage, read> physGx: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> physGy: array<vec2<f32>>;`
    : ''
  const gradientLoad = model.usesGradient
    ? `ux[f] = physGx[f * CELLS + i];
    uy[f] = physGy[f * CELLS + i];`
    : ''
  return /* wgsl */ `
struct Params {
${struct}
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> phys: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> physN: array<vec2<f32>>;
${gradientBindings}

const NF: u32 = ${fields}u;
const CELLS: u32 = ${cells}u;

fn nonlinear(u: array<vec2<f32>, ${fields}>, ux: array<vec2<f32>, ${fields}>, uy: array<vec2<f32>, ${fields}>, f: u32) -> vec2<f32> {
  ${model.nonlinearWgsl}
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = params;
  let i = gid.x;
  if (i >= CELLS) {
    return;
  }
  var u: array<vec2<f32>, ${fields}>;
  var ux: array<vec2<f32>, ${fields}>;
  var uy: array<vec2<f32>, ${fields}>;
  for (var f = 0u; f < NF; f++) {
    u[f] = phys[f * CELLS + i];
    ${gradientLoad}
  }
  for (var f = 0u; f < NF; f++) {
    physN[f * CELLS + i] = nonlinear(u, ux, uy, f);
  }
}
`
}
