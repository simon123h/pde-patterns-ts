import type { ModelImpl, ModelSpec, SpectralModel } from './types'

const inRange = (value: number, { min, max }: { min: number; max: number }) => value >= min && value <= max

/**
 * Joins a model's declarative spec with its implementation and checks that the two agree,
 * so that a mistake in a new model fails at start-up with a readable message.
 */
export function defineModel(spec: ModelSpec, impl: ModelImpl): SpectralModel {
  const fail = (message: string): never => {
    throw new Error(`model "${spec.id}": ${message}`)
  }
  const keys = new Set<string>()
  for (const p of spec.params) {
    if (!/^[A-Za-z_]\w*$/.test(p.key) || p.key === 'dx') fail(`invalid parameter key "${p.key}"`)
    if (keys.has(p.key)) fail(`duplicate parameter "${p.key}"`)
    keys.add(p.key)
    if (!inRange(p.default, p)) fail(`default of "${p.key}" is outside [${p.min}, ${p.max}]`)
  }
  if (!inRange(spec.domain.default, spec.domain)) fail('default domain size is outside its range')
  if (!inRange(spec.dt.default, spec.dt)) fail('default dt is outside its range')
  for (const view of spec.views) if (!Number.isInteger(view.field) || view.field < 0 || view.field >= spec.fields) fail(`view "${view.label}" shows missing field ${view.field}`)
  if (spec.views.length === 0) fail('no views')
  if (spec.paint && spec.paint.amounts.length !== spec.fields) fail('paint.amounts needs one entry per field')
  for (const preset of spec.presets ?? []) {
    for (const [key, value] of Object.entries(preset.params ?? {})) {
      const p = spec.params.find((q) => q.key === key)
      if (!p) fail(`preset "${preset.name}" sets unknown parameter "${key}"`)
      else if (!inRange(value, p)) fail(`preset "${preset.name}": ${key} = ${value} is outside [${p.min}, ${p.max}]`)
    }
    if (preset.domain !== undefined && !inRange(preset.domain, spec.domain)) fail(`preset "${preset.name}": domain size is outside its range`)
    if (preset.dt !== undefined && !inRange(preset.dt, spec.dt)) fail(`preset "${preset.name}": dt is outside its range`)
  }
  for (const [, key] of impl.nonlinearWgsl.matchAll(/\bparams\.(\w+)/g)) {
    if (key !== 'dx' && !keys.has(key!)) fail(`the WGSL uses unknown parameter "params.${key}"`)
  }
  return { ...spec, ...impl }
}
