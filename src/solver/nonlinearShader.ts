import type { SpectralModel } from '../models/types'
import { DERIVATIVES, derivativeCount } from './derivativeShader'

/** Generates the pointwise-nonlinearity kernel: physN[f] = N_f(phys, ∇phys) for all fields. */
export function nonlinearShader(model: SpectralModel, cells: number, nx: number): string {
  const real = !model.complex
  const T = real ? 'f32' : 'vec2<f32>'
  const load = (expr: string) => (real ? `vec2<f32>(${expr}, 0.0)` : expr)
  const fields = model.fields
  // the grid spacing is appended after the model parameters
  const struct = [...model.params.map((p) => p.key), 'dx'].map((key) => `  ${key}: f32,`).join('\n')
  const derived = model.usesGradient || model.usesHigherDerivatives ? derivativeCount(!!model.usesHigherDerivatives) : 0
  const derivativeBindings = DERIVATIVES.slice(0, derived)
    .map((name, i) => `@group(0) @binding(${3 + i}) var<storage, read> phys_${name}: array<${T}>;`)
    .join('\n')
  const derivativeLoad = DERIVATIVES.slice(0, derived)
    .map((name) => `${name}[f] = ${load(`phys_${name}[f * CELLS + i]`)};`)
    .join('\n    ')
  return /* wgsl */ `
struct Params {
${struct}
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> phys: array<${T}>;
@group(0) @binding(2) var<storage, read_write> physN: array<${T}>;
${derivativeBindings}

const NF: u32 = ${fields}u;
const CELLS: u32 = ${cells}u;
const NX: u32 = ${nx}u;

fn nonlinear(u: array<vec2<f32>, ${fields}>, ux: array<vec2<f32>, ${fields}>, uy: array<vec2<f32>, ${fields}>, lap: array<vec2<f32>, ${fields}>, lapx: array<vec2<f32>, ${fields}>, lapy: array<vec2<f32>, ${fields}>, bilap: array<vec2<f32>, ${fields}>, xy: vec2<f32>, f: u32) -> vec2<f32> {
  ${model.nonlinearWgsl}
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = params;
  let i = gid.x;
  if (i >= CELLS) {
    return;
  }
  let xy = vec2<f32>(f32(i % NX), f32(i / NX)) * params.dx;
  var u: array<vec2<f32>, ${fields}>;
  var ux: array<vec2<f32>, ${fields}>;
  var uy: array<vec2<f32>, ${fields}>;
  var lap: array<vec2<f32>, ${fields}>;
  var lapx: array<vec2<f32>, ${fields}>;
  var lapy: array<vec2<f32>, ${fields}>;
  var bilap: array<vec2<f32>, ${fields}>;
  for (var f = 0u; f < NF; f++) {
    u[f] = ${load('phys[f * CELLS + i]')};
    ${derivativeLoad}
  }
  for (var f = 0u; f < NF; f++) {
    physN[f * CELLS + i] = nonlinear(u, ux, uy, lap, lapx, lapy, bilap, xy, f)${real ? '.x' : ''};
  }
}
`
}
